#include "priority_queue_factory.hpp"
#include "thread_coordination.hpp"

#include "cxxopts.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef WITH_PAPI
extern "C" {
#include <papi.h>
#include <pthread.h>
}
#endif

using key_type = DefaultMinPriorityQueue::key_type;
using Handle = DefaultMinPriorityQueue::Handle;
using thread_coordination::timepoint_type;

#ifdef WITH_PAPI
// These are the event names used for the papi library to measure cache misses
// As these are hardware specific, you need to modify them or use generic PAPI events
static auto const L1d_cache_miss_event_name = "perf_raw::rc860";
static auto const L2_cache_miss_event_name = "perf_raw::r0864";

bool start_performance_counter(int& event_set) {
    if (int ret = PAPI_register_thread(); ret != PAPI_OK) {
        return false;
    }
    if (int ret = PAPI_create_eventset(&event_set); ret != PAPI_OK) {
        return false;
    }
    int event_code{};
    if (int ret = PAPI_event_name_to_code(L1d_cache_miss_event_name, &event_code); ret != PAPI_OK) {
        return false;
    }
    if (int ret = PAPI_add_event(event_set, event_code); ret != PAPI_OK) {
        return false;
    }
    if (int ret = PAPI_event_name_to_code(L2_cache_miss_event_name, &event_code); ret != PAPI_OK) {
        return false;
    }
    if (int ret = PAPI_add_event(event_set, event_code); ret != PAPI_OK) {
        return false;
    }
    if (int ret = PAPI_start(event_set); ret != PAPI_OK) {
        return false;
    }
    return true;
}
#endif

struct Settings {
    enum class WorkMode { Mixed, Split };
    enum class ElementDistribution { Uniform, Ascending, Descending };

    int num_threads = 4;
    std::size_t prefill_per_thread = 1 << 20;
    std::size_t elements_per_thread = 1 << 24;
    WorkMode work_mode = WorkMode::Mixed;
    int num_push_threads = 1;
    ElementDistribution element_distribution = ElementDistribution::Uniform;
    key_type min_key = 1;
    key_type max_key = key_type{1} << 30;
    int seed = 1;
#ifdef WITH_PAPI
    bool enable_performance_counter = false;
#endif

    bool set_work_mode(char c) {
        switch (c) {
            case 'm':
                work_mode = WorkMode::Mixed;
                return true;
            case 's':
                work_mode = WorkMode::Split;
                return true;
        }
        return false;
    }

    bool set_element_distribution(char c) {
        switch (c) {
            case 'u':
                element_distribution = ElementDistribution::Uniform;
                return true;
            case 'a':
                element_distribution = ElementDistribution::Ascending;
                return true;
            case 'd':
                element_distribution = ElementDistribution::Descending;
                return true;
        }
        return false;
    }

    [[nodiscard]] auto work_mode_str() const {
        switch (work_mode) {
            case WorkMode::Mixed:
                return "mixed";
            case WorkMode::Split:
                return "split";
        }
        return "unknown";
    }
    [[nodiscard]] auto element_distribution_str() const {
        switch (element_distribution) {
            case ElementDistribution::Uniform:
                return "uniform";
            case ElementDistribution::Ascending:
                return "ascending";
            case ElementDistribution::Descending:
                return "descending";
        }
        return "unknown";
    }

    [[nodiscard]] bool validate() const {
        if (num_threads <= 0) {
            return false;
        }
        if (min_key > max_key) {
            return false;
        }
        if (work_mode == WorkMode::Split &&
            ((num_push_threads < 0 || num_push_threads > num_threads) ||
             (num_push_threads == 0 && elements_per_thread > 0))) {
            return false;
        }
        if (work_mode == WorkMode::Split && num_push_threads == 0 && elements_per_thread > 0) {
            return false;
        }
        return true;
    }
};

struct Result {
    std::atomic<timepoint_type> start_time{timepoint_type::max()};
    std::atomic<timepoint_type> end_time{timepoint_type::min()};
    std::atomic<long long> num_failed_pops{0};
    std::atomic<long long> num_pops{0};
#ifdef WITH_PAPI
    std::atomic<long long> l1d_cache_misses{0};
    std::atomic<long long> l2_cache_misses{0};
#endif
#ifdef MQ_COUNT_STATS
    std::atomic<long long> num_locking_failed{0};
    std::atomic<long long> num_resets{0};
    std::atomic<long long> use_counts{0};
#endif

    void update_work_time(std::pair<timepoint_type, timepoint_type> const& work_time) {
        auto old_start = start_time.load(std::memory_order_relaxed);
        while (work_time.first < old_start &&
               !start_time.compare_exchange_weak(old_start, work_time.first, std::memory_order_relaxed)) {
        }
        auto old_end = start_time.load(std::memory_order_relaxed);
        while (work_time.second > old_end &&
               !end_time.compare_exchange_weak(old_end, work_time.second, std::memory_order_relaxed)) {
        }
    }
};

void execute_mixed(thread_coordination::Context const& ctx, Handle& handle, std::vector<key_type> const& keys,
                   Result& result) {
    long long num_failed_pops = 0;
    auto work_time = ctx.execute_synchronized_blockwise(
        keys.size(), [&handle, &keys, &num_failed_pops](auto start_index, auto count) {
            for (auto i = start_index; i < start_index + count; ++i) {
                handle.push({keys[i], keys[i]});
                DefaultMinPriorityQueue::value_type retval;
                while (!handle.try_pop(retval)) {
                    ++num_failed_pops;
                }
            }
        });
    result.num_failed_pops += num_failed_pops;
    result.update_work_time(work_time);
}

void execute_split_push(thread_coordination::Context const& ctx, Handle& handle, std::vector<key_type> const& keys,
                        Result& result) {
    auto work_time = ctx.execute_synchronized_blockwise(keys.size(), [&handle, &keys](auto start_index, auto count) {
        for (auto i = start_index; i < start_index + count; ++i) {
            handle.push({keys[i], keys[i]});
        }
    });
    result.update_work_time(work_time);
}

void execute_split_pop(thread_coordination::Context const& ctx, Handle& handle, Result& result,
                       std::size_t num_elements) {
    long long num_failed_pops = 0;
    auto work_time = ctx.execute_synchronized([&handle, &result, &num_failed_pops, num_elements]() {
        do {
            long long num_pops = 0;
            DefaultMinPriorityQueue::value_type retval;
            while (handle.try_pop(retval)) {
                ++num_pops;
            }
            ++num_failed_pops;
            if (num_pops == 0) {
                if (result.num_pops.load(std::memory_order_relaxed) >= static_cast<long long>(num_elements)) {
                    break;
                }
            } else {
                if (result.num_pops.fetch_add(num_pops, std::memory_order_relaxed) + num_pops >=
                    static_cast<long long>(num_elements)) {
                    break;
                }
            }
        } while (true);
    });
    assert(result.num_pops.load(std::memory_order_relaxed) == static_cast<long long>(num_elements));
    result.update_work_time(work_time);
    result.num_failed_pops += num_failed_pops;
}

template <typename RNG>
void generate_workload(Settings const& settings, std::vector<key_type>& keys, int id, RNG& rng) {
    auto start_index = static_cast<std::size_t>(id) * settings.elements_per_thread;
    switch (settings.element_distribution) {
        case Settings::ElementDistribution::Uniform: {
            auto key_dist = std::uniform_int_distribution<key_type>(settings.min_key, settings.max_key);
            for (std::size_t i = start_index; i < start_index + settings.elements_per_thread; ++i) {
                keys[i] = key_dist(rng);
            }
        } break;
        case Settings::ElementDistribution::Ascending: {
            auto range = settings.max_key - settings.min_key + 1;
            for (std::size_t i = start_index; i < start_index + settings.elements_per_thread; ++i) {
                keys[i] = settings.min_key + (i * range) / keys.size();
            }
        } break;
        case Settings::ElementDistribution::Descending: {
            auto range = settings.max_key - settings.min_key + 1;
            for (std::size_t i = start_index; i < start_index + settings.elements_per_thread; ++i) {
                keys[i] = settings.min_key + ((keys.size() - i - 1) * range) / keys.size();
            }
        } break;
    }
}

template <typename RNG>
void prefill(Settings const& settings, thread_coordination::Context ctx, Handle& handle, RNG& rng) {
    if (settings.prefill_per_thread > 0) {
        auto key_dist = std::uniform_int_distribution<key_type>(settings.min_key, settings.max_key);
        ctx.execute_synchronized([&handle, &key_dist, &rng, n = settings.prefill_per_thread]() {
            for (std::size_t i = 0; i < n; ++i) {
                auto key = key_dist(rng);
                handle.push({key, key});
            }
        });
    }
}

void benchmark_thread(thread_coordination::Context ctx, Settings const& settings, DefaultMinPriorityQueue& pq,
                      std::vector<key_type>& keys, Result& result) {
    std::seed_seq seed{settings.seed, ctx.get_id()};
    std::default_random_engine rng(seed);

    ctx.synchronize([]() { std::clog << "Generating keys..." << std::flush; });

    generate_workload(settings, keys, ctx.get_id(), rng);

    if (ctx.get_id() == 0) {
        std::clog << "done\nPrefilling..." << std::flush;
    }

    Handle handle = pq.get_handle(ctx.get_id());

    prefill(settings, ctx, handle, rng);

    if (ctx.get_id() == 0) {
        std::clog << "done\nWorking..." << std::flush;
    }

#ifdef MQ_COUNT_STATS
    handle.stats.reset();
#endif

#ifdef WITH_PAPI
    bool papi_started = false;
    int event_set = PAPI_NULL;
    if (settings.enable_performance_counter) {
        papi_started = start_performance_counter(event_set);
        if (!papi_started) {
            ctx.write(std::cerr) << "Failed to start counters\n";
        }
    }
#endif
    if (settings.work_mode == Settings::WorkMode::Mixed) {
        execute_mixed(ctx, handle, keys, result);
    } else {
        if (ctx.get_id() < settings.num_push_threads) {
            execute_split_push(ctx, handle, keys, result);
        } else {
            execute_split_pop(ctx, handle, result,
                              (settings.prefill_per_thread + settings.elements_per_thread) *
                                  static_cast<std::size_t>(settings.num_threads));
        }
    }
#ifdef WITH_PAPI
    if (papi_started) {
        std::array<long long, 2> counter{};
        if (int ret = PAPI_stop(event_set, counter.data()); ret != PAPI_OK) {
            ctx.write(std::cerr) << "Failed to stop counters\n";
        } else {
            result.l1d_cache_misses += counter[0];
            result.l2_cache_misses += counter[1];
        }
    }
#endif
#ifdef MQ_COUNT_STATS
    result.num_locking_failed += handle.stats.num_locking_failed;
    result.num_resets += handle.stats.num_resets;
    result.use_counts += handle.stats.use_counts;
#endif
    if (ctx.get_id() == 0) {
        std::clog << "done\n" << std::endl;
    }
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    std::clog << "Build type: Debug\n";
#else
    std::clog << "Build type: Release\n";
#endif
#ifdef WITH_PAPI
    std::clog << "Performance counter: enabled\n";
#else
    std::clog << "Performance counter: disabled\n";
#endif
    std::clog << "L1 cache linesize (bytes): " << L1_CACHE_LINESIZE << '\n';
    std::clog << "Pagesize (bytes): " << PAGESIZE << '\n';
    std::clog << "Priority queue: " << pq_name << '\n';

    std::clog << '\n';

    std::clog << "Command line:";
    for (int i = 0; i < argc; ++i) {
        std::clog << ' ' << argv[i];
    }
    std::clog << '\n' << '\n';

    Settings settings{};
    cxxopts::Options options(argv[0]);
    // clang-format off
    options.add_options()
        ("h,help", "Print this help")
        ("j,threads", "The number of threads", cxxopts::value<int>(settings.num_threads), "NUMBER")
        ("p,prefill", "The prefill per thread", cxxopts::value<std::size_t>(settings.prefill_per_thread), "NUMBER")
        ("n,keys", "The number of keys per thread", cxxopts::value<std::size_t>(settings.elements_per_thread), "NUMBER")
        ("w,work-mode", "Specify the work mode ([m]ixed, [s]plit)", cxxopts::value<char>(), "STRING")
        ("i,push-threads", "The number of pushing threads in split mode", cxxopts::value<int>(settings.num_push_threads), "NUMBER")
        ("e,element-distribution", "Specify the element distribution ([u]niform, [a]scending, [d]escending)", cxxopts::value<char>(), "STRING")
        ("l,min", "Specify the min key", cxxopts::value<key_type>(settings.min_key), "NUMBER")
        ("m,max", "Specify the max key", cxxopts::value<key_type>(settings.max_key), "NUMBER")
        ("s,seed", "Specify the initial seed", cxxopts::value<int>(settings.seed), "NUMBER")
#ifdef WITH_PAPI
        ("r,pc", "Enable Performance counter", cxxopts::value<>(settings.enable_performance_counter))
#endif
        ;
    // clang-format on
    add_pq_options(options);
    cxxopts::ParseResult args;
    try {
        args = options.parse(argc, argv);
    } catch (cxxopts::OptionParseException const& e) {
        std::cerr << "Error parsing arguments: " << e.what() << '\n';
        std::cerr << options.help() << std::endl;
        return 1;
    }
    if (args.count("help") > 0) {
        std::cerr << options.help() << std::endl;
        return 0;
    }
    if (args.count("work-mode") > 0) {
        auto work_mode = args["work-mode"].as<char>();
        if (!settings.set_work_mode(work_mode)) {
            std::cerr << "Invalid work mode: " << args["work-mode"].as<char>() << '\n';
            std::cerr << options.help() << std::endl;
            return 1;
        }
    }
    if (args.count("element-distribution") > 0) {
        auto element_distribution = args["element-distribution"].as<char>();
        if (!settings.set_element_distribution(element_distribution)) {
            std::cerr << "Invalid element distribution: " << args["element-distribution"].as<char>() << '\n';
            std::cerr << options.help() << std::endl;
            return 1;
        }
    }

    std::clog << "Threads: " << settings.num_threads << '\n'
              << "Prefill per thread: " << settings.prefill_per_thread << '\n'
              << "Elements per thread: " << settings.elements_per_thread << '\n'
              << "Operation mode: " << settings.work_mode_str();
    if (settings.work_mode == Settings::WorkMode::Split) {
        std::clog << " (" << settings.num_push_threads << " push)";
    }
    std::clog << '\n';
    std::clog << "Element distribution: " << settings.element_distribution_str() << '\n'
              << "Min key: " << settings.min_key << '\n'
              << "Max key: " << settings.max_key << '\n'
              << "Seed: " << settings.seed << '\n'
              << '\n';
    if (!settings.validate()) {
        std::cerr << "Invalid settings\n";
        std::cerr << options.help() << std::endl;
        return 1;
    }
#ifdef WITH_PAPI
    if (settings.enable_performance_counter) {
        if (int ret = PAPI_library_init(PAPI_VER_CURRENT); ret != PAPI_VER_CURRENT) {
            std::cerr << "Error initializing PAPI\n";
            return 1;
        }
        if (int ret = PAPI_thread_init((unsigned long (*)())(pthread_self)); ret != PAPI_OK) {
            std::cerr << "Error initializing threads for PAPI\n";
            return 1;
        }
        if (int ret = PAPI_query_named_event(L1d_cache_miss_event_name); ret != PAPI_OK) {
            std::cerr << "Cannot measure event '" << L1d_cache_miss_event_name << "'\n";
            return 1;
        }
        if (int ret = PAPI_query_named_event(L2_cache_miss_event_name); ret != PAPI_OK) {
            std::cerr << "Cannot measure event '" << L2_cache_miss_event_name << "'\n";
            return 1;
        }
    }
#endif

    Result result;
    {
        auto pq = create_pq<DefaultMinPriorityQueue>(
            settings.num_threads, settings.prefill_per_thread * static_cast<std::size_t>(settings.num_threads), args);

        std::vector<key_type> keys(static_cast<std::size_t>(settings.num_threads) * settings.elements_per_thread);
        thread_coordination::TaskHandle task_handle(settings.num_threads, benchmark_thread, settings, std::ref(pq),
                                                    std::ref(keys), std::ref(result));
        task_handle.wait();
    }

    std::clog << "Work time (s): " << std::setprecision(3)
              << std::chrono::duration<double>(result.end_time.load() - result.start_time.load()).count() << '\n';
    std::clog << "Failed pops: " << result.num_failed_pops << '\n';
#ifdef WITH_PAPI
    if (settings.enable_performance_counter) {
        std::clog << "L1d cache misses: " << result.l1d_cache_misses << '\n';
        std::clog << "L2 cache misses: " << result.l2_cache_misses << '\n';
    }
#endif
#ifdef MQ_COUNT_STATS
    std::clog << "Failed locks per operation: "
              << static_cast<double>(result.num_locking_failed) /
            static_cast<double>(static_cast<std::size_t>(settings.num_threads) * settings.elements_per_thread)
              << '\n';
    std::clog << "Average queue use count: "
              << static_cast<double>(result.use_counts) / static_cast<double>(result.num_resets) << '\n';
#endif

    std::cout << "# thread,prefill,elements,work-mode,push-threads,element-distribution,min-key,max-key,seed,work-time,"
                 "failed-pops,l1d-cache-misses,l2-cache-misses,num-resets-use-counts\n";
    std::cout << settings.num_threads << ',' << settings.prefill_per_thread << ',' << settings.elements_per_thread
              << ',' << settings.work_mode_str() << ',' << settings.num_push_threads << ','
              << settings.element_distribution_str() << ',' << settings.min_key << ',' << settings.max_key << ','
              << settings.seed << ',' << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(result.end_time.load() - result.start_time.load()).count() << ','
              << result.num_failed_pops;
#ifdef WITH_PAPI
    if (settings.enable_performance_counter) {
        std::cout << ',' << result.l1d_cache_misses << ',' << result.l2_cache_misses;
    } else {
        std::cout << ",n/a,n/a";
    }
#else
    std::cout << ",n/a,n/a";
#endif
#ifdef MQ_COUNT_STATS
    std::cout << ',' << result.num_resets << ',' << result.use_counts;
#else
    std::cout << ",n/a,n/a";
#endif
    std::cout << std::endl;
    return 0;
}
