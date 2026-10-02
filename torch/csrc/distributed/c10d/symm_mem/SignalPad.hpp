#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

#include <cstddef>
#include <cstdint>
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

// Layout of a signal pad whose user-visible part is `size` bytes, for a group
// of `world_size` ranks. Every index into a pad goes through these functions.
//
//   [0, size)        channels of world_size mailbox words, one per source rank:
//                    put_signal/wait_signal and the collectives' block
//                    barriers, which leave them zero. This is what
//                    get_signal_pad() returns.
//   u64 word         nccl_put_with_signal/nccl_wait_for_signal, which leave
//                    their value in place.
//   barrier state    per channel: an arrival slot per source rank, the
//                    multimem arrival counter, and the epoch. barrier() keeps
//                    state here that only moves forward, so a word left over
//                    by anything else cannot wedge it.
C10_HOST_DEVICE constexpr size_t signal_pad_channel_words(size_t world_size) {
  return world_size;
}

C10_HOST_DEVICE constexpr size_t signal_pad_num_channels(
    size_t size,
    size_t world_size) {
  return size / (sizeof(uint32_t) * signal_pad_channel_words(world_size));
}

C10_HOST_DEVICE constexpr size_t signal_pad_u64_word_offset(size_t size) {
  return (size + 7) / 8 * 8;
}

C10_HOST_DEVICE constexpr size_t signal_pad_barrier_state_offset(size_t size) {
  return signal_pad_u64_word_offset(size) + sizeof(unsigned long long);
}

C10_HOST_DEVICE constexpr size_t signal_pad_barrier_state_words(
    size_t world_size) {
  return world_size + 2;
}

constexpr size_t signal_pad_alloc_size(size_t size, size_t world_size) {
  return signal_pad_barrier_state_offset(size) +
      signal_pad_num_channels(size, world_size) *
      signal_pad_barrier_state_words(world_size) * sizeof(uint32_t);
}

} // namespace c10d::symmetric_memory
