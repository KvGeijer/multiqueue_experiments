/**
******************************************************************************
* @file:   select_queue.hpp
*
* @author: Marvin Williams
* @date:   2021/02/22 13:23
* @brief:
*******************************************************************************
**/
#pragma once
#ifndef UTILS_PRIORITY_QUEUE_FACTORY_HPP_INCLUDED
#define UTILS_PRIORITY_QUEUE_FACTORY_HPP_INCLUDED

#if defined PQ_CAPQ || defined PQ_CAPQ1 || defined PQ_CAPQ2 || defined PQ_CAPQ3 || defined PQ_CAPQ4
#include "capq.hpp"
#elif defined PQ_KLSM || defined PQ_KLSM256 || defined PQ_KLSM1024
#include "klsm.hpp"
#elif defined PQ_DLSM
#include "dlsm.hpp"
#elif defined PQ_LINDEN
#include "linden.hpp"
#elif defined PQ_SPRAYLIST
#include "spraylist.hpp"
#elif defined PQ_MQ_INT || defined PQ_MQ_INT_MERGING || defined PQ_MQ_INT_NB
#include "multiqueue/configurations.hpp"
#include "multiqueue/int_multiqueue.hpp"
#elif defined PQ_MQ_INT_AS
#include "multiqueue/configurations.hpp"
#include "multiqueue/int_multiqueue_assigned.hpp"
#else
#include "multiqueue/configurations.hpp"
#include "multiqueue/multiqueue.hpp"
#endif

#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <type_traits>

namespace util {

#if defined PQ_CAPQ || defined PQ_CAPQ1 || defined PQ_CAPQ2 || defined PQ_CAPQ3 || defined PQ_CAPQ4 ||      \
    defined PQ_KLSM || defined PQ_KLSM256 || defined PQ_KLSM1024 || defined PQ_DLSM || defined PQ_LINDEN || \
    defined PQ_SPRAYLIST

#define PQ_IS_WRAPPER 1

#else

#if defined PQ_MQ_NO_BUFFERING || defined PQ_MQ_INT_NB
using BaseConfig = multiqueue::configuration::NoBuffering;
#elif defined PQ_MQ_DELETE_BUFFERING
using BaseConfig = multiqueue::configuration::DeleteBuffering;
#elif defined PQ_MQ_INSERT_BUFFERING
using BaseConfig = multiqueue::configuration::InsertBuffering;
#elif defined PQ_MQ_FULL_BUFFERING || defined PQ_MQ_INT || defined PQ_MQ_INT_AS
using BaseConfig = multiqueue::configuration::FullBuffering;
#elif defined PQ_MQ_MERGING || defined PQ_MQ_INT_MERGING
using BaseConfig = multiqueue::configuration::Merging;
#else
#error No multiqueue variant selected
#endif

struct Config : BaseConfig {
#ifdef MQ_CONFIG_C
    static constexpr unsigned int C = MQ_CONFIG_C;
#endif
#ifdef MQ_CONFIG_K
    static constexpr unsigned int K = MQ_CONFIG_K;
#endif
#ifdef MQ_CONFIG_DELETION_BUFFER_SIZE
    static constexpr unsigned int DeletionBufferSize = MQ_CONFIG_DELETION_BUFFER_SIZE;
#endif
#ifdef MQ_CONFIG_INSERTION_BUFFER_SIZE
    static constexpr unsigned int InsertionBufferSize = MQ_CONFIG_INSERTION_BUFFER_SIZE;
#endif
#ifdef MQ_CONFIG_NODE_SIZE
    static constexpr unsigned int NodeSize = MQ_CONFIG_NODE_SIZE;
#endif
#ifdef MQ_CONFIG_HEAP_DEGREE
    static constexpr unsigned int HeapDegree = MQ_HEAP_DEGREE;
#endif
#ifdef MQ_CONFIG_NUMA
    static constexpr bool NumaFriendly = true;
#endif
#ifdef MQ_CONFIG_PHEROMONES
    static constexpr bool WithPheromones = true;
#endif
};
#endif

template <typename KeyType, typename ValueType>
struct PriorityQueueFactory {
#if defined PQ_CAPQ || defined PQ_CAPQ1
    using type = wrapper::capq<KeyType, ValueType, true, true, true>;
#elif defined PQ_CAPQ2
    using type = wrapper::capq<KeyType, ValueType, true, false, true>;
#elif defined PQ_CAPQ3
    using type = wrapper::capq<KeyType, ValueType, false, true, true>;
#elif defined PQ_CAPQ4
    using type = wrapper::capq<KeyType, ValueType, false, false, true>;
#elif defined PQ_KLSM || defined PQ_KLSM256
    using type = wrapper::klsm<KeyType, ValueType, 256>;
#elif defined PQ_KLSM1024
    using type = wrapper::klsm<KeyType, ValueType, 1024>;
#elif defined PQ_DLSM
    using type = wrapper::dlsm<KeyType, ValueType>;
#elif defined PQ_LINDEN
#elif defined PQ_SPRAYLIST
#elif defined PQ_MQ_INT || defined PQ_MQ_INT_MERGING || defined PQ_MQ_INT_NB
    using type = multiqueue::int_multiqueue<KeyType, ValueType, Config>;
#elif defined PQ_MQ_INT_AS
    using type = multiqueue::int_multiqueue_assigned<KeyType, ValueType, Config>;
#else
    using type = multiqueue::multiqueue<KeyType, ValueType, std::less<KeyType>, Config>;
#endif
};

template <>
struct PriorityQueueFactory<unsigned long, unsigned long> {
  using KeyType = unsigned long;
  using ValueType = unsigned long;
#if defined PQ_CAPQ || defined PQ_CAPQ1
    using type = wrapper::capq<KeyType, ValueType, true, true, true>;
#elif defined PQ_CAPQ2
    using type = wrapper::capq<KeyType, ValueType, true, false, true>;
#elif defined PQ_CAPQ3
    using type = wrapper::capq<KeyType, ValueType, false, true, true>;
#elif defined PQ_CAPQ4
    using type = wrapper::capq<KeyType, ValueType, false, false, true>;
#elif defined PQ_KLSM || defined PQ_KLSM256
    using type = wrapper::klsm<KeyType, ValueType, 256>;
#elif defined PQ_KLSM1024
    using type = wrapper::klsm<KeyType, ValueType, 1024>;
#elif defined PQ_DLSM
    using type = wrapper::dlsm<KeyType, ValueType>;
#elif defined PQ_LINDEN
    using type = wrapper::linden;
#elif defined PQ_SPRAYLIST
    using type = wrapper::spraylist;
#elif defined PQ_MQ_INT || defined PQ_MQ_INT_MERGING || defined PQ_MQ_INT_NB
    using type = multiqueue::int_multiqueue<KeyType, ValueType, Config>;
#elif defined PQ_MQ_INT_AS
    using type = multiqueue::int_multiqueue_assigned<KeyType, ValueType, Config>;
#else
    using type = multiqueue::multiqueue<KeyType, ValueType, std::less<KeyType>, Config>;
#endif
};

}  // namespace util

#endif  //! UTILS_PRIORITY_QUEUE_FACTORY_HPP_INCLUDED
