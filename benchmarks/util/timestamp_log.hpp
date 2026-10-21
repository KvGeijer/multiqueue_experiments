#pragma once

// Opt-in (LOG_TIMESTAMPS) history logging for relaxation monitoring.
//
// Wraps a priority queue handle so that every push and try_pop is bracketed by
// an invocation and a response timestamp. After the run, write() dumps all
// operations in the history format read by relaxation-monitoring:
//
//   enq <start> <end> <value>
//   deq <start> <end> <value>
//   empty <start> <end>
//
// The value of an element is its (key, value) pair packed into 64 bits as
// (key << 32) | value, so both must fit in 32 bits (the run aborts otherwise). As
// the monitor treats the value as the priority (min-queue), equal keys are ordered
// by the second component. For this to identify elements, the pair must be unique:
// the node id in dijkstra and the element id in monotonic.
//
// The queue itself is given the packed value as its key, so that it orders by exactly
// what the monitor orders by, and equal keys no longer tie. The handle unpacks the
// key again on pop, so the benchmark still sees the key it pushed.

#include <cxxopts.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace timestamp_log {

inline std::uint64_t get_next_timestamp() noexcept {
    std::uint64_t cycles_low;
    std::uint64_t cycles_high;
    asm volatile(
        "RDTSCP\n\t"
        "mov %%rdx , %0\n\t"
        "mov %%rax , %1\n\t"
        "LFENCE\n\t"
        : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rcx", "%rdx");
    return (cycles_high << 32) | cycles_low;
}

inline void compiler_barrier() noexcept {
    asm volatile("" ::: "memory");
}

enum class Op : std::uint8_t { push, pop, empty };

struct Record {
    std::uint64_t inv;
    std::uint64_t res;
    std::uint64_t packed;
    Op op;
};

inline std::filesystem::path& output_file() {
    static std::filesystem::path path = "history.log";
    return path;
}

inline void register_cmd_options(cxxopts::Options& cmd) {
    cmd.add_options()("history-file", "Output file for the timestamped operation history",
                      cxxopts::value<std::filesystem::path>(output_file()), "PATH");
}

// Fail early rather than after a long run
inline void check_output() {
    if (std::ofstream(output_file()).fail()) {
        std::cerr << "Error: Could not open file " << output_file() << " for writing\n";
        std::exit(EXIT_FAILURE);
    }
}

// Owns the per-handle logs so that handles stay cheap to move and may be destroyed before write()
class Registry {
   public:
    using Log = std::vector<Record>;

   private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<Log>> logs_;

   public:
    static Registry& instance() {
        static Registry registry;
        return registry;
    }

    Log* create() {
        auto log = std::make_unique<Log>();
        log->reserve(1 << 16);
        std::lock_guard<std::mutex> lock{mutex_};
        logs_.push_back(std::move(log));
        return logs_.back().get();
    }

    // Must not be called while handles are still in use
    void write(std::ostream& out) {
        std::uint64_t origin = UINT64_MAX;
        for (auto const& log : logs_) {
            for (auto const& r : *log) {
                origin = std::min(origin, r.inv);
            }
        }
        out << "# pq\n";
        out << "# timestamps: rdtscp cycles relative to the first invocation\n";
        out << "# value: (key << 32) | value\n";
        for (auto const& log : logs_) {
            for (auto const& r : *log) {
                auto start = r.inv - origin;
                auto end = r.res - origin;
                switch (r.op) {
                    case Op::push:
                        out << "enq " << start << ' ' << end << ' ' << r.packed << '\n';
                        break;
                    case Op::pop:
                        out << "deq " << start << ' ' << end << ' ' << r.packed << '\n';
                        break;
                    case Op::empty:
                        out << "empty " << start << ' ' << end << '\n';
                        break;
                }
            }
        }
    }
};

// Wraps a handle whose value_type is a (key, value) pair. Implicitly constructible from the wrapped handle
// so that it can be dropped in by changing the handle_type alias.
template <typename Inner>
class Handle {
   public:
    using value_type = typename Inner::value_type;

   private:
    Inner handle_;
    Registry::Log* log_;

    static std::uint64_t pack(value_type const& v) {
        auto key = static_cast<std::uint64_t>(v.first);
        auto value = static_cast<std::uint64_t>(v.second);
        if (key > UINT32_MAX || value > UINT32_MAX) {
            std::cerr << "Error: Element (" << key << ", " << value << ") does not fit in 32 bits per component\n";
            std::exit(EXIT_FAILURE);
        }
        return (key << 32) | value;
    }

    // The element as stored in the queue: the packed value is its key
    static value_type to_inner(std::uint64_t packed, value_type const& v) {
        return value_type{static_cast<typename value_type::first_type>(packed), v.second};
    }

    // The element as the benchmark pushed it: the key is the upper half of the packed key
    static value_type from_inner(value_type const& v) {
        return value_type{static_cast<typename value_type::first_type>(static_cast<std::uint64_t>(v.first) >> 32),
                          v.second};
    }

   public:
    Handle(Inner&& handle) : handle_{std::move(handle)}, log_{Registry::instance().create()} {
    }

    decltype(auto) push(value_type const& value) {
        auto packed = pack(value);
        auto const inner_value = to_inner(packed, value);
        compiler_barrier();
        auto inv = get_next_timestamp();
        compiler_barrier();
        if constexpr (std::is_void_v<decltype(handle_.push(inner_value))>) {
            handle_.push(inner_value);
            compiler_barrier();
            auto res = get_next_timestamp();
            compiler_barrier();
            log_->push_back({inv, res, packed, Op::push});
        } else {
            auto success = handle_.push(inner_value);
            compiler_barrier();
            auto res = get_next_timestamp();
            compiler_barrier();
            // A rejected push never enters the queue and is not part of the history
            if (success) {
                log_->push_back({inv, res, packed, Op::push});
            }
            return success;
        }
    }

    std::optional<value_type> try_pop() {
        compiler_barrier();
        auto inv = get_next_timestamp();
        compiler_barrier();
        auto result = handle_.try_pop();
        compiler_barrier();
        auto res = get_next_timestamp();
        compiler_barrier();
        if (!result) {
            log_->push_back({inv, res, 0, Op::empty});
            return std::nullopt;
        }
        log_->push_back({inv, res, static_cast<std::uint64_t>(result->first), Op::pop});
        return from_inner(*result);
    }
};

inline void write_history(std::filesystem::path const& path) {
    std::ofstream out(path);
    Registry::instance().write(out);
}

}  // namespace timestamp_log
