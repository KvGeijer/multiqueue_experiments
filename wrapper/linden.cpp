#include "linden.hpp"

extern "C" {
#include "spraylist_linden/gc/gc.h"
#include "spraylist_linden/linden.h"
}
#undef min
#undef max

#include <cstddef>
#include <iostream>
#include <utility>

namespace wrapper {

void Linden::pq_deleter::operator()(pq_t* p) {
    // Avoid segfault
    ::insert(p, 1, 1);
    pq_destroy(p);
    _destroy_gc_subsystem();
}

Linden::Linden(unsigned int /* num_threads */) {
    _init_gc_subsystem();
    pq_.reset(pq_init(32));
}

void Linden::Handle::push(value_type const& value) const {
    ::insert(pq_, value.first + 1, value.second);
}

bool Linden::Handle::try_pop(value_type& retval) const {
    retval.second = ::deletemin_key(pq_, &retval.first);
    if (retval.first == sentinel_) {
        return false;
    }
    --retval.first;
    return true;
}

Linden::Handle Linden::get_handle() {
    auto h = Handle{};
    h.pq_ = pq_.get();
    return h;
}

void Linden::push(value_type const& value) const {
    ::insert(pq_.get(), value.first + 1, value.second);
}

bool Linden::try_pop(value_type& retval) const {
    retval.second = ::deletemin_key(pq_.get(), &retval.first);
    if (retval.first == sentinel_) {
        return false;
    }
    --retval.first;
    return true;
}

}  // namespace wrapper
