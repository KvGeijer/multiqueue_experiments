#include "external/cxxopts.hpp"

#include "system_config.hpp"
#include "utils/inserting_strategy.hpp"
#include "utils/priority_queue_factory.hpp"
#include "utils/thread_coordination.hpp"
#include "utils/threading.hpp"

#include <time.h>
#include <x86intrin.h>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined THROUGHPUT && !defined QUALITY
#define THROUGHPUT
#endif

#if defined THROUGHPUT
#undef QUALITY
#elif defined QUALITY
#undef THROUGHPUT
#else
#error No test mode specified
#endif

using key_type = unsigned long;
using value_type = unsigned long;
static_assert(std::is_unsigned_v<value_type>, "Value type must be unsigned");
using PriorityQueue =
    typename util::PriorityQueueFactory<key_type, value_type>::type;

using namespace std::chrono_literals;

#ifdef QUALITY

using clock_type = std::chrono::steady_clock;
using time_point_type = clock_type::time_point;
using tick_type = std::uint64_t;

static inline time_point_type get_time_point() noexcept {
  return clock_type::now();
}

static inline tick_type get_tick_steady() noexcept {
  return static_cast<tick_type>(clock_type::now().time_since_epoch().count());
}

static inline tick_type get_tick_realtime() noexcept {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<tick_type>(ts.tv_sec * 1000000000 + ts.tv_nsec);
}

static inline tick_type get_tick_rdtsc() noexcept { return __rdtsc(); }

#endif

struct Settings {
  std::size_t prefill_size = 1'000'000;
  std::chrono::nanoseconds sleep_between_operations = 0ns;
#ifdef THROUGHPUT
  std::chrono::milliseconds test_duration = 3s;
#else
  std::size_t min_num_delete_operations = 10'000'000;
#endif
  unsigned int num_threads = 4;
  std::uint32_t seed;
  InsertConfig<key_type> insert_config{
      InsertPolicy::Uniform,
      KeyDistribution::Uniform,
      std::numeric_limits<value_type>::min(),
      std::numeric_limits<value_type>::max() - 3,  // Some pqs use sentinels
      1,
      100,
  };
};

#ifdef QUALITY

static constexpr unsigned int bits_for_thread_id = 8;
static constexpr value_type value_mask =
    (static_cast<value_type>(1)
     << (std::numeric_limits<value_type>::digits - bits_for_thread_id)) -
    1;

static constexpr value_type to_value(unsigned int thread_id,
                                     value_type elem_id) noexcept {
  return (static_cast<value_type>(thread_id)
          << (std::numeric_limits<value_type>::digits - bits_for_thread_id)) |
         (elem_id & value_mask);
}

static constexpr unsigned int get_thread_id(value_type value) noexcept {
  return static_cast<unsigned int>(
      value >> (std::numeric_limits<value_type>::digits - bits_for_thread_id));
}

static constexpr value_type get_elem_id(value_type value) noexcept {
  return value & value_mask;
}

struct InsertionLogEntry {
  tick_type tick;
  key_type key;
};

struct DeletionLogEntry {
  tick_type tick;
  value_type value;
};

std::vector<InsertionLogEntry>* insertions;
std::vector<DeletionLogEntry>* deletions;
std::vector<tick_type>* failed_deletions;

#else

// Used to guarantee writing of result so it can't be optimized out;
struct alignas(2 * L1_CACHE_LINESIZE) DummyResult {
  volatile key_type key;
  volatile value_type value;
};

DummyResult* dummy_result;

#endif

std::uint32_t* thread_seeds;

std::atomic_size_t num_insertions;
std::atomic_size_t num_deletions;
std::atomic_size_t num_failed_deletions;

std::atomic_bool start_flag;

#ifdef THROUGHPUT
std::atomic_bool stop_flag;
#endif

#ifdef QUALITY
std::atomic_size_t num_delete_operations;
#endif

// Assume rdtsc is thread-safe and synchronized on each CPU
// Assumption false

struct Task {
  static void run(thread_coordination::Context ctx, PriorityQueue& pq,
                  Settings const& settings) {
#ifdef QUALITY
    std::vector<InsertionLogEntry> local_insertions;
    local_insertions.reserve(settings.prefill_size +
                             settings.min_num_delete_operations);
    std::vector<DeletionLogEntry> local_deletions;
    local_deletions.reserve(settings.min_num_delete_operations);
    std::vector<tick_type> local_failed_deletions;
    local_failed_deletions.reserve(1'000'000);
#endif
    std::size_t num_local_insertions = 0;
    std::size_t num_local_deletions = 0;
    std::size_t num_local_failed_deletions = 0;

#ifdef PQ_SPRAYLIST
    pq.init_thread(ctx.get_num_threads());
#endif
    std::seed_seq seq{thread_seeds[ctx.get_id()]};
    auto gen = std::mt19937(seq);
    auto dist = std::uniform_int_distribution<long>(
        0, settings.sleep_between_operations.count());

    unsigned int stage = 0;

    auto handle = pq.get_handle(ctx.get_id());

    auto inserter = InsertingStrategy<key_type>{
        ctx.get_id(), settings.insert_config, thread_seeds[ctx.get_id()] + 1};

    if (ctx.is_main()) {
      if (settings.prefill_size > 0) {
        std::clog << "Prefilling..." << std::flush;
        for (size_t i = 0; i < settings.prefill_size; ++i) {
          key_type const key = inserter.get_key();
#ifdef QUALITY
          value_type const value = to_value(
              ctx.get_id(), static_cast<value_type>(local_insertions.size()));
          local_insertions.push_back(InsertionLogEntry{0, key});
#else
          value_type const value = key;
#endif
          pq.push(handle, {key, value});
        }
        std::clog << "done" << std::endl;
      }
    }
    ctx.synchronize(stage++, [&ctx]() {
      std::clog << "Starting the stress test..." << std::flush;
      ctx.notify_coordinator();
    });
    while (!start_flag.load(std::memory_order_relaxed)) {
      _mm_pause();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    std::pair<key_type, value_type> retval;
#ifdef THROUGHPUT
    while (!stop_flag.load(std::memory_order_relaxed)) {
#else
    while (num_delete_operations.load(std::memory_order_relaxed) <
           settings.min_num_delete_operations) {
#endif
      if (inserter.insert()) {
        key_type const key = inserter.get_key();
#ifdef QUALITY
        value_type const value = to_value(
            ctx.get_id(), static_cast<value_type>(local_insertions.size()));
        pq.push(handle, {key, value});
        _mm_lfence();
        auto tick = get_tick_realtime();
        _mm_lfence();
        local_insertions.push_back(InsertionLogEntry{tick, key});
#else
        pq.push(handle, {key, key});
#endif
        ++num_local_insertions;
      } else {
        bool success = pq.extract_top(handle, retval);
#ifdef QUALITY
        _mm_lfence();
        auto tick = get_tick_realtime();
        _mm_lfence();
        if (success) {
          local_deletions.push_back(DeletionLogEntry{tick, retval.second});
          num_delete_operations.fetch_add(1, std::memory_order_relaxed);
        } else {
          local_failed_deletions.push_back(tick);
          ++num_local_failed_deletions;
        }
#else
        if (success) {
          dummy_result[ctx.get_id()].key = retval.first;
          dummy_result[ctx.get_id()].value = retval.second;
        } else {
          ++num_local_failed_deletions;
        }
#endif
        ++num_local_deletions;
      }
      if (settings.sleep_between_operations > 0us) {
        std::this_thread::sleep_for(std::chrono::nanoseconds{dist(gen)});
      }
    }
    ctx.synchronize(stage++, []() { std::clog << "done" << std::endl; });
#ifdef QUALITY
    insertions[ctx.get_id()] = std::move(local_insertions);
    deletions[ctx.get_id()] = std::move(local_deletions);
    failed_deletions[ctx.get_id()] = std::move(local_failed_deletions);
#endif

    num_insertions += num_local_insertions;
    num_deletions += num_local_deletions;
    num_failed_deletions += num_local_failed_deletions;
  }

  static threading::thread_config get_config(
      thread_coordination::Context const& ctx) {
    threading::thread_config config;
    config.cpu_set.reset();
    config.cpu_set.set(ctx.get_id());
    return config;
  }
};

int main(int argc, char* argv[]) {
  Settings settings{};

  cxxopts::Options options("performance test",
                           "This executable measures and records the "
                           "performance of relaxed priority queues");
  // clang-format off
    options.add_options()
      ("n,prefill", "Specify the number of elements to prefill the queue with "
       "(default: 1'000'000)", cxxopts::value<size_t>(), "NUMBER")
      ("i,insert", "Specify the insert policy as one of \"uniform\", \"split\", \"producer\", \"alternating\" "
       "(default: uniform)", cxxopts::value<std::string>(), "ARG")
      ("j,threads", "Specify the number of threads "
       "(default: 4)", cxxopts::value<unsigned int>(), "NUMBER")
      ("w,sleep", "Specify the sleep time between operations in ns"
       "(default: 0)", cxxopts::value<unsigned int>(), "NUMBER")
      ("d,distribution", "Specify the key distribution as one of \"uniform\", \"dijkstra\", \"ascending\", \"descending\", \"threadid\" "
       "(default: uniform)", cxxopts::value<std::string>(), "ARG")
      ("m,max", "Specify the max key "
       "(default: MAX)", cxxopts::value<key_type>(), "NUMBER")
      ("l,min", "Specify the min key "
       "(default: 0)", cxxopts::value<key_type>(), "NUMBER")
      ("s,seed", "Specify the initial seed"
       "(default: 0)", cxxopts::value<std::uint32_t>(), "NUMBER")
#ifdef THROUGHPUT
      ("t,time", "Specify the test timeout in ms "
       "(default: 3000)", cxxopts::value<unsigned int>(), "NUMBER")
#else
      ("o,deletions", "Specify the minimum number of deletions "
       "(default: 10'000'000)", cxxopts::value<std::size_t>(), "NUMBER")
#endif
      ("h,help", "Print this help");
  // clang-format on

  try {
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cerr << options.help() << std::endl;
      return 0;
    }
    if (result.count("prefill") > 0) {
      settings.prefill_size = result["prefill"].as<size_t>();
    }
    if (result.count("insert") > 0) {
      std::string policy = result["insert"].as<std::string>();
      if (policy == "uniform") {
        settings.insert_config.insert_policy = InsertPolicy::Uniform;
      } else if (policy == "split") {
        settings.insert_config.insert_policy = InsertPolicy::Split;
      } else if (policy == "producer") {
        settings.insert_config.insert_policy = InsertPolicy::Producer;
      } else if (policy == "alternating") {
        settings.insert_config.insert_policy = InsertPolicy::Alternating;
      } else {
        std::cerr << "Unknown insert policy \"" << policy << "\"\n";
        return 1;
      }
    }
    if (result.count("threads") > 0) {
      settings.num_threads = result["threads"].as<unsigned int>();
    }
    if (result.count("sleep") > 0) {
      settings.sleep_between_operations =
          std::chrono::nanoseconds{result["sleep"].as<unsigned int>()};
    }
    if (result.count("distribution") > 0) {
      std::string dist = result["distribution"].as<std::string>();
      if (dist == "uniform") {
        settings.insert_config.key_distribution = KeyDistribution::Uniform;
      } else if (dist == "ascending") {
        settings.insert_config.key_distribution = KeyDistribution::Ascending;
      } else if (dist == "descending") {
        settings.insert_config.key_distribution = KeyDistribution::Descending;
      } else if (dist == "dijkstra") {
        settings.insert_config.key_distribution = KeyDistribution::Dijkstra;
      } else if (dist == "threadid") {
        settings.insert_config.key_distribution = KeyDistribution::ThreadId;
      } else {
        std::cerr << "Unknown key distribution \"" << dist << "\"\n";
        return 1;
      }
    }
    if (result.count("max") > 0) {
      settings.insert_config.max_key = result["max"].as<key_type>();
    }
    if (result.count("min") > 0) {
      settings.insert_config.min_key = result["min"].as<key_type>();
    }
#ifdef THROUGHPUT
    if (result.count("time") > 0) {
      settings.test_duration =
          std::chrono::milliseconds{result["time"].as<unsigned int>()};
    }
#else
    if (result.count("deletions") > 0) {
      settings.min_num_delete_operations =
          result["deletions"].as<std::size_t>();
    }
#endif
    if (result.count("seed") > 0) {
      settings.seed = result["seed"].as<std::uint32_t>();
    }
  } catch (cxxopts::OptionParseException const& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifndef NDEBUG
  std::clog << "Using debug build!\n\n";
#endif

#ifdef QUALITY
  if (settings.num_threads > (1 << bits_for_thread_id) - 1) {
    std::cerr << "Too many threads, increase the number of thread bits!"
              << std::endl;
    return 1;
  }
#endif

#if defined THROUGHPUT
  std::clog << "Measuring throughput!\n\n";
#elif defined QUALITY
  std::clog << "Recording quality log!\n\n";
#endif

  std::clog << "Settings: \n\t"
            << "Prefill size: " << settings.prefill_size << "\n\t"
#ifdef THROUGHPUT
            << "Test duration: " << settings.test_duration.count() << " ms\n\t"
#else
            << "Min deletions: " << settings.min_num_delete_operations << "\n\t"
#endif
            << "Sleep between operations: "
            << settings.sleep_between_operations.count() << " ns\n\t"
            << "Threads: " << settings.num_threads << "\n\t"
            << "Insert policy: "
            << get_insert_policy_name(settings.insert_config.insert_policy)
            << "\n\t"
            << "Min key: " << settings.insert_config.min_key << "\n\t"
            << "Max key: " << settings.insert_config.max_key << "\n\t"
            << "Key distribution: "
            << get_key_distribution_name(
                   settings.insert_config.key_distribution)
            << "\n\t"
            << "Dijkstra min increase: "
            << settings.insert_config.dijkstra_min_increase << "\n\t"
            << "Dijkstra max increase: "
            << settings.insert_config.dijkstra_max_increase << "\n\t"
            << "Seed: " << settings.seed;
  std::clog << "\n\n";

  std::clog << "Using priority queue: " << PriorityQueue::description() << '\n';
#ifdef PQ_IS_WRAPPER
  PriorityQueue pq{settings.num_threads};
#else
  PriorityQueue pq{settings.num_threads, settings.seed};
#endif

#ifdef QUALITY
  insertions = new std::vector<InsertionLogEntry>[settings.num_threads];
  deletions = new std::vector<DeletionLogEntry>[settings.num_threads];
  failed_deletions = new std::vector<tick_type>[settings.num_threads];
  num_delete_operations = 0;
#else
  dummy_result = new DummyResult[settings.num_threads];
#endif
  std::seed_seq seq{settings.seed + 1};
  thread_seeds = new std::uint32_t[settings.num_threads];
  seq.generate(thread_seeds, thread_seeds + settings.num_threads);
  num_insertions = 0;
  num_deletions = 0;
  num_failed_deletions = 0;
  start_flag.store(false, std::memory_order_relaxed);
#ifdef THROUGHPUT
  stop_flag.store(false, std::memory_order_relaxed);
#endif
  std::atomic_thread_fence(std::memory_order_release);
  thread_coordination::ThreadCoordinator coordinator{settings.num_threads};
  coordinator.run<Task>(std::ref(pq), settings);
  coordinator.wait_until_notified();
  start_flag.store(true, std::memory_order_release);
#ifdef THROUGHPUT
  std::this_thread::sleep_for(settings.test_duration);
  stop_flag.store(true, std::memory_order_release);
#endif
  coordinator.join();
#ifdef THROUGHPUT
  std::cout << "Insertions: " << num_insertions
            << "\nDeletions: " << num_deletions
            << "\nFailed deletions: " << num_failed_deletions
            << "\nOps/s: " << std::fixed << std::setprecision(1)
            << (1000.0 * static_cast<double>(num_insertions + num_deletions)) /
                   static_cast<double>(settings.test_duration.count())
            << std::endl;

  delete[] dummy_result;
#else
  std::cout << settings.num_threads << '\n';
  for (unsigned int t = 0; t < settings.num_threads; ++t) {
    for (auto const& [tick, key] : insertions[t]) {
      std::cout << "i " << t << ' ' << tick << ' ' << key << '\n';
    }
  }

  for (unsigned int t = 0; t < settings.num_threads; ++t) {
    for (auto const& [tick, value] : deletions[t]) {
      std::cout << "d " << t << ' ' << tick << ' ' << get_thread_id(value)
                << ' ' << get_elem_id(value) << '\n';
    }
  }

  for (unsigned int t = 0; t < settings.num_threads; ++t) {
    for (auto tick : failed_deletions[t]) {
      std::cout << "f " << t << ' ' << tick << '\n';
    }
  }

  std::cout << std::flush;

  delete[] insertions;
  delete[] deletions;
  delete[] failed_deletions;
#endif
  delete[] thread_seeds;
  return 0;
}
