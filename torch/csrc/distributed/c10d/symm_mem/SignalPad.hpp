#pragma once

#include <c10/util/intrusive_ptr.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace c10d::symmetric_memory {

// The signal pad of one (process group, device): the synchronization state
// every built-in symmetric-memory operation of the group uses on that device,
// whichever allocation it runs on. Ops sharing a pad then share a group, so
// GroupStreamGuard's per-group ordering is exactly what keeps them apart.
//
// Immutable once built and shared by every handle of the group. `owner` keeps
// the mapped memory alive; how is the backend's business.
class SignalPad {
 public:
  SignalPad(
      c10::intrusive_ptr<c10::intrusive_ptr_target> owner,
      std::vector<void*> peers,
      void** peers_dev,
      void* multicast,
      size_t size)
      : owner_(std::move(owner)),
        peers_(std::move(peers)),
        peers_dev_(peers_dev),
        multicast_(multicast),
        size_(size) {}

  // Each rank's pad, indexed by group rank.
  const std::vector<void*>& peers() const {
    return peers_;
  }
  // The same pointers in a device array of size world_size.
  void** peers_dev() const {
    return peers_dev_;
  }
  // This rank's view of the multicast mapping, or nullptr without multicast.
  void* multicast() const {
    return multicast_;
  }
  // Bytes available to channels and to get_signal_pad(). Fixed when the pad is
  // created; set_signal_pad_size() only affects pads created later.
  size_t size() const {
    return size_;
  }

 private:
  c10::intrusive_ptr<c10::intrusive_ptr_target> owner_;
  std::vector<void*> peers_;
  void** peers_dev_;
  void* multicast_;
  size_t size_;
};

// A pad allocation is the channel area, `size` bytes of world_size words per
// channel, which get_signal_pad() returns; then one u64 word for
// nccl_put_with_signal/nccl_wait_for_signal, which leave their value in place;
// then the multimem barrier's arrival counters, one word per channel. Neither
// shares a word with a mailbox, and the channel area keeps the layout kernels
// outside PyTorch index.
constexpr size_t signal_pad_u64_word_offset(size_t size) {
  return (size + 7) / 8 * 8;
}

constexpr size_t signal_pad_barrier_counters_offset(size_t size) {
  return signal_pad_u64_word_offset(size) + sizeof(unsigned long long);
}

constexpr size_t signal_pad_alloc_size(size_t size, size_t world_size) {
  return signal_pad_barrier_counters_offset(size) +
      size / (sizeof(uint32_t) * world_size) * sizeof(uint32_t);
}

} // namespace c10d::symmetric_memory
