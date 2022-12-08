#include "priority_queue_factory.hpp"
#include "thread_coordination.hpp"
#ifdef PQ_MQ
#include "multiqueue/config.hpp"
#endif

#include "cxxopts.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
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

#ifdef USE_PAPI
extern "C" {
#include <papi.h>
#include <pthread.h>
}
#endif

using key_type = unsigned long;
using value_type = unsigned long;

using PriorityQueue = util::priority_queue_type<key_type, value_type, true>;

static constexpr std::size_t DefaultPrefillPerThread = 1 << 20;
static constexpr std::size_t DefaultOperationsPerThread = 1 << 24;
static constexpr unsigned int DefaultNumThreads = 4;
static constexpr double DefaultPopProbability = 0.5;
static constexpr key_type DefaultMaxKey = key_type{1} << 30;

struct Settings {
    std::size_t prefill_per_thread = DefaultPrefillPerThread;
    std::size_t operations_per_thread = DefaultOperationsPerThread;
    unsigned int num_threads = DefaultNumThreads;
    double pop_prob = DefaultPopProbability;
    std::uint32_t seed = 1;
    key_type min_key = 1;
    key_type max_key = DefaultMaxKey;
    bool no_work = false;
};

class Benchmark {
   public:
    using Handle = typename PriorityQueue::handle_type;

    using tick_type = std::uint64_t;

    class Operation {
        key_type data;

       public:
        static constexpr key_type PopOp = 0;

        Operation() : data{PopOp} {
        }

        Operation(key_type insert_key) : data{insert_key} {
            assert(insert_key != PopOp);
        }

        [[nodiscard]] constexpr bool is_pop() const noexcept {
            return data == PopOp;
        }
        [[nodiscard]] constexpr key_type get_insert_key() const noexcept {
            assert(!is_pop());
            return data;
        }
    };
    struct Data {
        std::vector<Operation> operations;
        thread_coordination::duration_type prefill_time{};
        thread_coordination::duration_type work_time{};
        std::atomic_uint num_failed_pops = 0;
        std::atomic_llong cache_loads = 0;
        std::atomic_llong cache_load_misses = 0;
#ifdef MULTIQUEUE_COUNT_STATS
        std::atomic_size_t num_locking_failed{0};
        std::atomic_size_t num_resets{0};
#endif
        std::mutex m;
    };

   private:
#ifdef USE_PAPI
    static bool start_performance_counter(int& event_set) {
        if (int ret = PAPI_register_thread(); ret != PAPI_OK) {
            return false;
        }
        if (int ret = PAPI_create_eventset(&event_set); ret != PAPI_OK) {
            return false;
        }
        if (int ret = PAPI_add_named_event(event_set, "perf::L1-DCACHE-LOADS"); ret != PAPI_OK) {
            return false;
        }
        if (int ret = PAPI_add_named_event(event_set, "perf::L1-DCACHE-LOAD-MISSES"); ret != PAPI_OK) {
            return false;
        }
        if (int ret = PAPI_start(event_set); ret != PAPI_OK) {
            return false;
        }
        return true;
    }
#endif

    static void work(thread_coordination::Context const& ctx, Handle& handle, Data& data) {
        unsigned int num_failed_pops = 0;
#ifdef USE_PAPI
        int event_set = PAPI_NULL;
        bool papi_started = start_performance_counter(event_set);
        if (!papi_started) {
            ctx.write(std::cerr) << "Failed to start counters\n";
        }
#endif
        ctx.execute_synchronized_blockwise_timed(data.work_time, data.operations.begin(), data.operations.end(),
                                                 [&handle, &num_failed_pops](auto block_begin, auto block_end) {
                                                     for (auto it = block_begin; it != block_end; ++it) {
                                                         if (it->is_pop()) {
                                                             PriorityQueue::value_type retval;
                                                             // Let retval escape
                                                             asm volatile("" ::"g"(&retval));
                                                             if (!handle.try_pop(retval)) {
                                                                 ++num_failed_pops;
                                                             }
                                                             // "Use" memory to force write to retval
                                                             asm volatile("" ::: "memory");
                                                         } else {
                                                             key_type key = it->get_insert_key();
                                                             handle.push({key, key});
                                                         }
                                                     }
                                                 });
#ifdef USE_PAPI
        if (papi_started) {
            long long cache_stats[2];
            if (int ret = PAPI_stop(event_set, cache_stats); ret != PAPI_OK) {
                ctx.write(std::cerr) << "Failed to stop counters\n";
            } else {
                data.cache_loads += cache_stats[0];
                data.cache_load_misses += cache_stats[1];
            }
        }
#endif
        data.num_failed_pops += num_failed_pops;
    }

   public:
    static void run(thread_coordination::Context ctx, Settings const& settings, Data& data, PriorityQueue& pq) {
        if (ctx.get_id() == 0) {
            std::clog << "Generating operations..." << std::flush;
        }

        std::seed_seq seed{settings.seed, static_cast<std::uint32_t>(ctx.get_id())};
        std::default_random_engine rng(seed);
        auto key_dist = std::uniform_int_distribution<key_type>(settings.min_key, settings.max_key);
        auto pop_dist = std::bernoulli_distribution(settings.pop_prob);

        // Prefill
        std::vector<key_type> prefill_keys(settings.prefill_per_thread);
        std::generate(prefill_keys.begin(), prefill_keys.end(), [&key_dist, &rng]() { return key_dist(rng); });

        // Workload
        auto block_begin =
            data.operations.begin() + ctx.get_id() * static_cast<std::ptrdiff_t>(settings.operations_per_thread);
        std::generate_n(block_begin, settings.operations_per_thread,
                        [&]() { return pop_dist(rng) ? Operation{} : Operation{key_dist(rng)}; });

        ctx.synchronize([]() { std::clog << "done\nPrefilling..." << std::flush; });

        Handle handle = [&data, &pq]() {
            std::scoped_lock l{data.m};
            return pq.get_handle();
        }();

        ctx.execute_synchronized_timed(data.prefill_time, [&handle, &prefill_keys]() {
            for (auto k : prefill_keys) {
                handle.push({k, k});
            }
        });

        if (ctx.get_id() == 0) {
            std::clog << "done\nWorking..." << std::flush;
        }
#ifdef MULTIQUEUE_COUNT_STATS
        handle.stats.num_locking_failed = 0;
        handle.stats.num_resets = 0;
#endif
        if (!settings.no_work) {
            work(ctx, handle, data);
        }
#ifdef MULTIQUEUE_COUNT_STATS
        data.num_locking_failed += handle.stats.num_locking_failed;
        data.num_resets += handle.stats.num_resets;
#endif
        if (ctx.get_id() == 0) {
            std::clog << "done\n" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    std::cout << "Build type: Debug\n";
#else
    std::cout << "Build type: Release\n";
#endif
#ifdef USE_PAPI
    std::cout << "Performance counter: enabled\n";
#else
    std::cout << "Performance counter: disabled\n";
#endif
    std::cout << "L1 cache linesize (bytes): " << L1_CACHE_LINESIZE << '\n';
    std::cout << "Pagesize (bytes): " << PAGESIZE << '\n';
    std::cout << '\n';

    Settings settings;
#ifdef PQ_MQ
    multiqueue::Config mq_config;
#endif

    std::filesystem::path out_file;

    cxxopts::Options options(argv[0]);
    options.add_options()
        // clang-format off
        ("h,help", "Print this help")
        ("j,threads", "The number of threads", cxxopts::value<unsigned int>(settings.num_threads), "NUMBER")
        ("p,prefill", "The prefill per thread", cxxopts::value<std::size_t>(settings.prefill_per_thread), "NUMBER")
        ("n,ops", "The number of operations per thread", cxxopts::value<std::size_t>(settings.operations_per_thread), "NUMBER")
        ("d,pop-prob", "Specify the probability of pops in [0,1]", cxxopts::value<double>(settings.pop_prob), "NUMBER")
        ("l,min", "Specify the min key", cxxopts::value<key_type>(settings.min_key), "NUMBER")
        ("m,max", "Specify the max key", cxxopts::value<key_type>(settings.max_key), "NUMBER")
        ("s,seed", "Specify the initial seed", cxxopts::value<std::uint32_t>(settings.seed), "NUMBER")
#ifdef PQ_MQ
        ("c,factor", "The factor for queues", cxxopts::value<unsigned int>(mq_config.c), "NUMBER")
        ("k,stickiness", "The stickiness period", cxxopts::value<unsigned int>(mq_config.stickiness), "NUMBER")
#endif
        ("o,outfile", "Output data in csv (comma-separated)", cxxopts::value<std::filesystem::path>(out_file), "PATH")
        ("x,no-work", "Don't perform the actual benchmark", cxxopts::value<bool>(settings.no_work), "PATH")
        // clang-format on
        ;

    try {
        auto result = options.parse(argc, argv);
        if (result.count("help") > 0) {
            std::cerr << options.help() << std::endl;
            return 0;
        }
    } catch (cxxopts::OptionParseException const& e) {
        std::cerr << "Error parsing arguments: " << e.what() << '\n';
        std::cerr << options.help() << std::endl;
        return 1;
    }

    std::cout << "Command line:";
    for (int i = 0; i < argc; ++i) {
        std::cout << ' ' << argv[i];
    }
    std::cout << '\n';
    std::cout << "Threads: " << settings.num_threads << '\n'
              << "Prefill per thread: " << settings.prefill_per_thread << '\n'
              << "Operations per thread: " << settings.operations_per_thread << '\n'
              << "Pop probability: " << std::fixed << std::setprecision(2) << settings.pop_prob << '\n'
              << "Min key: " << settings.min_key << '\n'
              << "Max key: " << settings.max_key << '\n'
              << "Seed: " << settings.seed << '\n'
#ifdef PQ_MQ
              << "Factor: " << mq_config.c << '\n'
              << "Stickiness: " << mq_config.stickiness << '\n'
#endif
              << '\n';

#ifdef USE_PAPI
    if (int ret = PAPI_library_init(PAPI_VER_CURRENT); ret != PAPI_VER_CURRENT) {
        std::cerr << "Error initializing PAPI\n";
        return 1;
    }
    if (int ret = PAPI_thread_init((unsigned long (*)())(pthread_self)); ret != PAPI_OK) {
        std::cerr << "Error initializing threads for PAPI\n";
        return 1;
    }
    if (int ret = PAPI_query_named_event("perf::L1-DCACHE-LOADS"); ret != PAPI_OK) {
        std::cerr << "Cannot measure cache loads\n";
        return 1;
    }
    if (int ret = PAPI_query_named_event("perf::L1-DCACHE-LOAD-MISSES"); ret != PAPI_OK) {
        std::cerr << "Cannot measure cache misses\n";
        return 1;
    }
#endif

#ifdef PQ_MQ
    mq_config.seed = settings.seed;
    auto pq = PriorityQueue(settings.num_threads, mq_config);
#else
    auto pq = PriorityQueue(settings.num_threads);
#endif

    std::cout << "Data structure: ";
    util::describe::describe(std::cout, pq);
    std::cout << '\n';

    Benchmark::Data benchmark_data{};
    benchmark_data.operations.resize(static_cast<std::size_t>(settings.num_threads) * settings.operations_per_thread);

    thread_coordination::TaskHandle<Benchmark> task_handle(settings.num_threads, std::cref(settings),
                                                           std::ref(benchmark_data), std::ref(pq));

    task_handle.join();

    std::cout << "Prefill time (s): " << std::setprecision(3)
              << std::chrono::duration<double>(benchmark_data.prefill_time).count() << '\n';
    std::cout << "Work time (s): " << std::setprecision(3)
              << std::chrono::duration<double>(benchmark_data.work_time).count() << '\n';
    std::cout << "Failed pops: " << benchmark_data.num_failed_pops << '\n';
#ifdef USE_PAPI
    std::cout << "Cache loads/misses: " << benchmark_data.cache_loads << '/' << benchmark_data.cache_load_misses
              << '\n';
#else
    std::cout << "Cache loads/misses: 0/0\n";
#endif
#ifdef MULTIQUEUE_COUNT_STATS
    std::cout << "Failed locks per operation: "
              << static_cast<double>(benchmark_data.num_locking_failed) /
            static_cast<double>(settings.num_threads * settings.operations_per_thread)
              << '\n';
    std::cout << "Operations per reset: "
              << static_cast<double>(settings.num_threads * settings.operations_per_thread) /
            static_cast<double>(benchmark_data.num_resets)
              << '\n';
#endif

    if (!out_file.empty()) {
        auto out = std::ofstream{out_file};
        if (!out.is_open()) {
            std::cerr << "Could not open file to write out benchmark_datas\n";
            return 1;
        }
        out << "threads,prefill,operations,pop_prob,min_key,max_key,"
               "seed,prefill_time,work_time,failed_pops,cache_loads,cache_load_misses\n";
        out << settings.num_threads << ',' << settings.prefill_per_thread << ',' << settings.operations_per_thread
            << ',' << settings.pop_prob << ',' << settings.min_key << ',' << settings.max_key << ',' << settings.seed
            << ',' << std::fixed << std::setprecision(3)
            << std::chrono::duration<double>(benchmark_data.prefill_time).count() << ',' << std::fixed
            << std::setprecision(3) << std::chrono::duration<double>(benchmark_data.work_time).count() << ','
            << benchmark_data.num_failed_pops
#ifdef USE_PAPI
            << ',' << benchmark_data.cache_loads << ',' << benchmark_data.cache_load_misses
#else
            << ",0,0"
#endif
            << std::endl;
    }
    return 0;
}