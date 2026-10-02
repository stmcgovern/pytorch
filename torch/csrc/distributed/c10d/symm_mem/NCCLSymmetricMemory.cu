#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <algorithm>
#include <vector_types.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/error.h>
#include <mutex>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>

namespace c10d {
namespace symmetric_memory {

/* Start of NCCLAllocation implementation */

static StoreExchange storeExchange = StoreExchange("NCCLAllocation");

struct NCCLAllocation {
  // The ncclMemAlloc base, which alloc() hands back.
  void* alloc_base;
  // Size of the user-visible data buffer in bytes, as requested by alloc().
  size_t buffer_size;
  int device_idx;
  std::mutex mutex;
  // Map of group name to peer alloc info
  ska::flat_hash_map<std::string, c10::intrusive_ptr<NCCLPeerAllocInfo>>
      peer_alloc_infos_;

  NCCLAllocation(void* alloc_base, size_t buffer_size, int device_idx)
      : alloc_base(alloc_base),
        buffer_size(buffer_size),
        device_idx(device_idx) {}

  ~NCCLAllocation() {
    if (should_skip_cuda_cleanup(device_idx)) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_idx);
    // Single free for the combined buffer + signal pad region.
    ncclResult_t res = ncclMemFree(alloc_base);
    if (res != ncclSuccess) {
        LOG(WARNING) << "ncclMemFree failed in NCCLAllocation dtor: "
                      << ncclGetErrorString(res);
    }
  }
};

namespace {

// Base allocation ptr -> owning NCCL allocation metadata.
using NCCLAllocMap = ska::flat_hash_map<void*, std::unique_ptr<NCCLAllocation>>;
// (Tensor storage/data ptr, group name) -> cached SymmetricMemory handle.
using NCCLSymmMemMap = ska::flat_hash_map<
    SymmMemKey,
    c10::intrusive_ptr<NCCLSymmetricMemory>,
    SymmMemKeyHash>;
// Base allocation ptr -> cached `(tensor ptr, group)` keys derived from it.
using NCCLSymmMemKeysByAlloc =
    ska::flat_hash_map<void*, ska::flat_hash_set<SymmMemKey, SymmMemKeyHash>>;

bool pointer_in_allocation(void* ptr, const NCCLAllocation& allocation) {
  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
  auto buffer_ptr = reinterpret_cast<uintptr_t>(allocation.alloc_base);
  return ptr_int >= buffer_ptr && ptr_int < buffer_ptr + allocation.buffer_size;
}

NCCLAllocMap::iterator find_allocation_covering_linear(
    void* ptr,
    NCCLAllocMap& allocations) {
  return std::find_if(
      allocations.begin(),
      allocations.end(),
      [&](const auto& entry) {
        return pointer_in_allocation(ptr, *entry.second);
      });
}

NCCLAllocMap::iterator find_allocation_covering(
    void* ptr,
    NCCLAllocMap& allocations) {
  auto alloc_it = allocations.find(ptr);
  if (alloc_it != allocations.end()) {
    return alloc_it;
  }
  // `ptr` is not an allocation key (a MemPool hands out interior pointers), so
  // scan for the allocation whose [buffer, buffer + size) range covers it.
  // TODO: this linear std::find_if is O(n) in the number of live allocations.
  // Make it O(log n) by switching NCCLAllocMap to an ordered map and using
  // upper_bound to find the covering allocation.
  return find_allocation_covering_linear(ptr, allocations);
}

} // namespace

// Before NCCL 2.29, we can use device-side APIs to get peer pointers.
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
static __global__ void build_ptr_dev(
  ncclWindow_t  handle,
  void**  buffers,        // out: peer buffer pointers
  int  world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int peer = tid; peer < world_size; peer += stride) {
      buffers[peer] = ncclGetLsaPointer(handle, 0, peer);
  }
}
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#endif // NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)

class NCCLPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  NCCLPeerAllocInfo(
      NCCLAllocation* allocation,
      std::string group_name)
      : buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx),
        group_name_(std::move(group_name))
  {
    c10::cuda::CUDAGuard guard(device_idx_);
    auto group = resolve_process_group(group_name_);
    rank_ = group->getRank();
    world_size_ = group->getSize();
    // Look up the host ncclComm by group name in NCCLDevCommManager. Any
    // backend that owns a NCCL-compatible communicator (ProcessGroupNCCL, or
    // an external library exposing its ncclComm — torchcomms is one such
    // example) publishes into this registry at comm-init time, so symm_mem
    // doesn't need to know which backend the PG is wrapping.
    auto& mgr = NCCLDevCommManager::get(
        c10::Device(c10::DeviceType::CUDA, device_idx_));
    comm_ = mgr.get_comm(group_name_);
    TORCH_CHECK(
        comm_ != nullptr,
        "NCCL symmetric memory: NCCLDevCommManager returned a null comm for "
        "group '",
        group_name_,
        "'. If you are using ProcessGroups, please make sure its backend has "
        "been eagerly initialized by filling `device_id` in the "
        "`init_process_group` call.");

    const size_t total_size = at::round_up(buffer_size_, 16UL);
    C10D_NCCL_CHECK(
      ncclCommWindowRegister(comm_, allocation->alloc_base, total_size, &combined_win_, NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register segment with ptr ",
          allocation->alloc_base,
          ", size ",
          total_size,
          " on rank ",
          rank_));

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    // (Host comm is already published into NCCLDevCommManager by the
    // owning backend at comm-init time. The earlier mgr.get_comm() call
    // above relied on that. No re-register here.)

    // Starting from NCCL 2.28, we can get peer pointers.
    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    buffers_.resize(world_size_);

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
    // Lack of host-side API to get peer pointers, so a kernel writes them and
    // copies the results to host.
    int threads = std::min(128, world_size_);
    auto stream = at::cuda::getCurrentCUDAStream();
    build_ptr_dev<<<1, threads, 0, stream>>>(
        combined_win_, buffers_dev_, world_size_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    C10_CUDA_CHECK(cudaMemcpy(
      buffers_.data(),  // dst (host)
      buffers_dev_,  // src (device)
      arr_size,
      cudaMemcpyDeviceToHost));
#else
  // Starting from NCCL 2.29, we can use host-side APIs to get peer pointers.
  for (int i = 0; i < world_size_; i++) {
    // If peer is not accessible within LSA domain, `ncclGetPeerDevicePointer`
    // returns nullptr.
    C10D_NCCL_CHECK(
      ncclGetPeerDevicePointer(combined_win_, 0, i, &buffers_[i]),
      "ncclGetPeerDevicePointer failed");
  }
  C10_CUDA_CHECK(cudaMemcpy(
    buffers_dev_,  // dst (device)
    buffers_.data(),  // src (host)
    arr_size,
    cudaMemcpyHostToDevice));

  // Starting from NCCL 2.29, we can use `ncclGetLsaMultimemDevicePointer`
  // to get multicast address.
  void* mc_addr = nullptr;
  // Skip CHECK on purpose to improve fault tolerance since some machine's
  // Fabric Manager may be in bad NVLink Sharp state.
  if (ncclGetLsaMultimemDevicePointer(combined_win_, 0, &mc_addr) ==
          ncclSuccess &&
      mc_addr != nullptr) {
    mc_addr_ = mc_addr;
  }
#endif // NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  }

  // Exact copy is not needed / supported
  NCCLPeerAllocInfo(const NCCLPeerAllocInfo& other) = delete;
  NCCLPeerAllocInfo& operator=(const NCCLPeerAllocInfo& other) = delete;
  NCCLPeerAllocInfo(NCCLPeerAllocInfo&& other) = default;
  NCCLPeerAllocInfo& operator=(NCCLPeerAllocInfo&& other) = default;

  ~NCCLPeerAllocInfo() {
    if (should_skip_cuda_cleanup(device_idx_)) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_idx_);
    if (combined_win_ != nullptr) {
      auto res = ncclCommWindowDeregister(comm_, combined_win_);
      if (res != ncclSuccess) {
        LOG(WARNING) << "ncclCommWindowDeregister failed: "
                     << ncclGetErrorString(res);
      }
    }
    if (buffers_dev_ != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(buffers_dev_);
    }
  }

 private:
  size_t buffer_size_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  void** buffers_dev_{nullptr};
  std::string group_name_;
  ncclWindow_t combined_win_{nullptr};
  // Multicast address (data buffer base within the multicast mapping)
  void* mc_addr_{nullptr};
  ncclComm_t comm_{nullptr};
  // The group's pad, set by rendezvous() before the info is shared with any
  // handle. Unset only on the info that maps a pad itself.
  std::shared_ptr<const SignalPad> pad_;
  friend class NCCLSymmetricMemory;
  friend class NCCLSymmetricMemoryAllocator;
};

NCCLSymmetricMemory::NCCLSymmetricMemory(
    c10::intrusive_ptr<NCCLPeerAllocInfo> pai,
    size_t offset)
    : pai_(std::move(pai)),
      offset_(offset),
      rank_(pai_->rank_),
      world_size_(pai_->world_size_),
      device_idx_(pai_->device_idx_) {
  TORCH_INTERNAL_ASSERT(offset_ < pai_->buffer_size_, "offset out of range");
  TORCH_INTERNAL_ASSERT(pai_->pad_ != nullptr, "handle without a signal pad");
}

std::vector<void*> NCCLSymmetricMemory::get_buffer_ptrs() {
  return pai_->buffers_;
}

std::vector<void*> NCCLSymmetricMemory::get_signal_pad_ptrs() {
  return pai_->pad_->peers();
}

void** NCCLSymmetricMemory::get_buffer_ptrs_dev() {
  return pai_->buffers_dev_;
}

void** NCCLSymmetricMemory::get_signal_pad_ptrs_dev() {
  return pai_->pad_->peers_dev();
}

size_t NCCLSymmetricMemory::get_buffer_size() {
  return pai_->buffer_size_;
}

size_t NCCLSymmetricMemory::get_signal_pad_size() {
  return pai_->pad_->size();
}

bool NCCLSymmetricMemory::has_multicast_support() {
  return pai_->mc_addr_ != nullptr;
}

void* NCCLSymmetricMemory::get_multicast_ptr() {
  if (!has_multicast_support()) {
    return nullptr;
  }
  return static_cast<char*>(pai_->mc_addr_) + offset_;
}

void NCCLSymmetricMemory::barrier(int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  TORCH_CHECK(
      pai_->pad_->peers_dev() != nullptr,
      "NCCLSymmetricMemory::barrier requires peer signal pad pointers, which "
      "are only populated when peers are accessible over the symmetric-memory "
      "(LSA/NVLink) domain.");
  check_channel(channel, world_size_, get_signal_pad_size());
  c10::cuda::CUDAGuard device_guard(device_idx_);
  GroupStreamGuard stream_guard(pai_->group_name_);
  barrier_kernel<<<
      1,
      std::max(at::cuda::warp_size(), world_size_),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->pad_->peers_dev()),
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NYI");
#endif
}

void NCCLSymmetricMemory::put_signal(int dst_rank, int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_ONE_SIDED_API
  check_rank(dst_rank, world_size_);
  TORCH_CHECK(channel == 0, "channel must be 0 (sigIdx is reserved for future use)");

  c10::cuda::CUDAGuard guard(device_idx_);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto& manager = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
  ncclComm_t comm = manager.get_comm(pai_->group_name_);

  // use ncclSignal for pure signaling without data transfer
  C10D_NCCL_CHECK(
      ncclSignal(
          dst_rank,
          channel,
          0,
          0,
          comm,
          stream),
      c10::str("ncclSignal failed for dst_rank=", dst_rank, ", channel=", channel));
#else
  TORCH_CHECK(false, "NYI");
#endif
}

void NCCLSymmetricMemory::wait_signal(int src_rank, int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_ONE_SIDED_API
  check_rank(src_rank, world_size_);
  TORCH_CHECK(channel == 0, "channel must be 0 (sigIdx is reserved for future use)");

  c10::cuda::CUDAGuard guard(device_idx_);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto& manager = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
  ncclComm_t comm = manager.get_comm(pai_->group_name_);

  // create signal descriptor for waiting - populate all fields
  ncclWaitSignalDesc_t signalDesc;
  signalDesc.opCnt = 1;
  signalDesc.peer = src_rank;
  signalDesc.sigIdx = channel;
  signalDesc.ctx = 0;

  C10D_NCCL_CHECK(
      ncclWaitSignal(
          1,
          &signalDesc,
          comm,
          stream),
      c10::str("ncclWaitSignal failed for src_rank=", src_rank, ", channel=", channel));
#else
  TORCH_CHECK(false, "NYI");
#endif
}

int NCCLSymmetricMemory::get_rank() {
  return rank_;
}

int NCCLSymmetricMemory::get_world_size() {
  return world_size_;
}

c10::Device NCCLSymmetricMemory::get_device() {
  return c10::Device(c10::DeviceType::CUDA, device_idx_);
}

ncclWindow_t NCCLSymmetricMemory::get_window() {
  return pai_->combined_win_;
}

size_t NCCLSymmetricMemory::get_offset() {
  return offset_;
}

size_t NCCLSymmetricMemory::get_window_offset() {
  return offset_;
}

#ifdef NCCL_HAS_HOST_CFT
// Both CFT queries only succeed once NCCL has created the communicator's
// logical endpoints. With `hostCftMode` enabled that happens during the first
// `ncclCommWindowRegister`, i.e. during rendezvous; otherwise the caller would
// have had to build a CFT-enabled `ncclDevComm` first.
static constexpr const char* kHostCftHint =
    "Host-side CFT requires CFT-capable hardware and a process group whose "
    "communicator was created with `host_cft_mode` enabled "
    "(ProcessGroupNCCL.NCCLConfig.host_cft_mode).";
#endif // NCCL_HAS_HOST_CFT

NCCLCftHandle NCCLSymmetricMemory::get_peer_cft_handle(int peer) {
#ifdef NCCL_HAS_HOST_CFT
  TORCH_CHECK(
      peer >= 0 && peer < world_size_,
      "NCCLSymmetricMemory::get_peer_cft_handle: invalid peer ",
      peer);
  c10::cuda::CUDAGuard guard(device_idx_);
  ncclCftLeId le_id = 0;
  size_t le_offset = 0;
  C10D_NCCL_CHECK(
      ncclGetPeerDeviceLeInfo(
          pai_->combined_win_, get_window_offset(), peer, &le_id, &le_offset),
      c10::str(
          "ncclGetPeerDeviceLeInfo failed for peer ", peer, ". ", kHostCftHint));
  return NCCLCftHandle{le_id, le_offset};
#else
  TORCH_CHECK(
      false, "NCCL host-side CFT is not supported. Requires NCCL >= 2.31.2");
#endif
}

NCCLCftHandle NCCLSymmetricMemory::get_multimem_cft_handle() {
#ifdef NCCL_HAS_HOST_CFT
  // Unlike the unicast query, this one may still have to bind the multicast
  // team (and barrier over the group) if the endpoint wasn't created eagerly.
  c10::cuda::CUDAGuard guard(device_idx_);
  ncclCftLeId le_id = 0;
  size_t le_offset = 0;
  C10D_NCCL_CHECK(
      ncclGetMultimemDeviceLeInfo(
          pai_->combined_win_, get_window_offset(), &le_id, &le_offset),
      c10::str("ncclGetMultimemDeviceLeInfo failed. ", kHostCftHint));
  return NCCLCftHandle{le_id, le_offset};
#else
  TORCH_CHECK(
      false, "NCCL host-side CFT is not supported. Requires NCCL >= 2.31.2");
#endif
}

std::string NCCLSymmetricMemory::get_group_name() {
  return pai_->group_name_;
}

namespace {

// Owns a group signal pad: its memory and the window over it. Members are
// destroyed in reverse order, so the window goes before the memory.
struct NCCLSignalPadOwner : public c10::intrusive_ptr_target {
  std::unique_ptr<NCCLAllocation> allocation;
  c10::intrusive_ptr<NCCLPeerAllocInfo> pai;
};

// What a rank tells its peers before a pad is allocated.
struct PadRequest {
  size_t size;
  uint8_t has_comm;
};

} // namespace

class NCCLSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "NCCLSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name");

    c10::cuda::CUDAGuard guard(device_idx);
    // ncclMemAlloc rejects a zero size.
    const size_t total_size = at::round_up(std::max<size_t>(size, 1), 16UL);
    void* alloc_base;
    C10D_NCCL_CHECK(ncclMemAlloc(&alloc_base, total_size), "ncclMemAlloc");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.emplace(
          alloc_base,
          std::make_unique<NCCLAllocation>(alloc_base, size, device_idx));
    }
    return alloc_base;
  }

  void free(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto alloc_it = allocations_.find(ptr);
    if (alloc_it == allocations_.end()) {
      return;
    }
    auto cache_keys_it = symm_mem_keys_by_alloc_.find(ptr);
    if (cache_keys_it != symm_mem_keys_by_alloc_.end()) {
      for (const auto& key : cache_keys_it->second) {
        symm_mems_.erase(key);
      }
      symm_mem_keys_by_alloc_.erase(cache_keys_it);
    }
    allocations_.erase(alloc_it);
  };

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      TORCH_CHECK(
          false, ptr, " is not allocated with NCCLSymmetricMemoryAllocator");
    }
    return it->second->buffer_size;
  };

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(group_name.has_value(), "group_name must be provided");
    NCCLAllocation* allocation;
    // The covering allocation's map key, the base alloc() returned.
    void* buffer_ptr_key = nullptr;
    SymmMemKey key{ptr, *group_name};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = symm_mems_.find(key);
      if (it != symm_mems_.end()) {
        return it->second;
      }

      // Find the allocation covering the ptr under the allocator lock.
      // We grab a raw pointer to the NCCLAllocation so we can release the
      // allocator lock before doing expensive per-allocation work.
      auto alloc_it = find_allocation_covering(ptr, allocations_);
      TORCH_CHECK(
          alloc_it != allocations_.end(),
          "Pointer not within any SymmetricMemory allocation, "
          "is the tensor allocated from SymmetricMemory?");
      allocation = alloc_it->second.get();
      buffer_ptr_key = alloc_it->first;
    }

    // The pad first: creating it on the group's first rendezvous registers a
    // window of its own, which must not overlap this one's.
    auto pad = signal_pad(*group_name, allocation->device_idx);

    // Get or create peer alloc info for the group under the per-allocation
    // lock. This serializes concurrent rendezvous on the same allocation
    // for different groups (e.g., forward vs backward).
    std::lock_guard<std::mutex> alloc_lock(allocation->mutex);
    auto& peer_alloc_infos = allocation->peer_alloc_infos_;
    auto& pai = peer_alloc_infos[*group_name];
    if (!pai) {
      auto info = c10::make_intrusive<NCCLPeerAllocInfo>(allocation, *group_name);
      info->pad_ = std::move(pad);
      pai = std::move(info);
    }
    size_t offset = reinterpret_cast<uintptr_t>(ptr) -
        reinterpret_cast<uintptr_t>(buffer_ptr_key);
    // Create the SymmetricMemory handle.
    auto symm_mem = c10::make_intrusive<NCCLSymmetricMemory>(pai, offset);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Insert the SymmetricMemory handle into the map (cache), keyed by the
      // (Tensor storage ptr, group name) pair.
      auto [it, inserted] = symm_mems_.emplace(key, symm_mem);
      if (!inserted) {
        // This condition should rarely happen, only when another thread happens
        // to be concurrently rendezvousing with the same allocation for the
        // same group.  For safety, we return the existing SymmetricMemory
        // handle and discard the new one.
        return it->second;
      }
      // There is no more use of `key`; we can move it into the per-allocation
      // key set to avoid an extra copy. Key by the data pointer (the value
      // returned by alloc()), matching the lookup done in free().
      symm_mem_keys_by_alloc_[buffer_ptr_key].insert(std::move(key));
    }
    return symm_mem;
  }

  bool has_multicast_support(int device_idx) override {
    return device_has_multicast_support(device_idx);
  }

  bool has_allocation(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return find_allocation_covering(ptr, allocations_) != allocations_.end();
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CUDA;
  }

  std::string name() override {
    return "NCCL";
  }

 private:
  // The signal pad of (group, device), created by the group's first
  // rendezvous on the device. It is an allocation of its own, kept out of
  // allocations_ so it is never taken for a user tensor.
  std::shared_ptr<const SignalPad> signal_pad(
      const std::string& group_name,
      int device_idx) {
    auto group = resolve_process_group(group_name);
    return get_or_create_signal_pad(
        group, static_cast<c10::DeviceIndex>(device_idx), [&] {
          const size_t size = get_signal_pad_size();
          // Agree before allocating. Registering the window is collective, so
          // a rank that cannot take part would leave its peers waiting in it;
          // every rank sees the same answers and fails together instead.
          const PadRequest mine{
              size,
              NCCLDevCommManager::get(
                  c10::Device(c10::DeviceType::CUDA, device_idx))
                  .has_comm(group_name)};
          auto reqs = storeExchange.all_gather(
              group->getStore(), group->getRank(), group->getSize(), mine);
          for (size_t r = 0; r < reqs.size(); ++r) {
            TORCH_CHECK(
                reqs[r].has_comm,
                "NCCL symmetric memory: rank ",
                r,
                " has no NCCL communicator for group '",
                group_name,
                "'. Initialize the process group's NCCL backend eagerly by "
                "passing `device_id` to `init_process_group`.");
            TORCH_CHECK(
                reqs[r].size == size,
                "NCCL symmetric memory: ranks of group '",
                group_name,
                "' disagree on the signal pad size (",
                reqs[r].size,
                " vs ",
                size,
                " bytes). Call set_signal_pad_size() with the same value on "
                "every rank.");
          }
          c10::cuda::CUDAGuard guard(device_idx);
          const size_t alloc_size = signal_pad_alloc_size(size, group->getSize());
          void* base = nullptr;
          C10D_NCCL_CHECK(ncclMemAlloc(&base, alloc_size), "ncclMemAlloc");
          auto owner = c10::make_intrusive<NCCLSignalPadOwner>();
          owner->allocation = std::make_unique<NCCLAllocation>(
              base, alloc_size, device_idx);
          C10_CUDA_CHECK(cudaMemset(base, 0, alloc_size));
          owner->pai = c10::make_intrusive<NCCLPeerAllocInfo>(
              owner->allocation.get(), group_name);
          const auto& pai = *owner->pai;
          return std::make_shared<const SignalPad>(
              owner, pai.buffers_, pai.buffers_dev_, pai.mc_addr_, size);
        });
  }

  std::mutex mutex_;
  NCCLAllocMap allocations_;
  NCCLSymmMemMap symm_mems_;
  NCCLSymmMemKeysByAlloc symm_mem_keys_by_alloc_;
};

struct RegisterNCCLSymmetricMemoryAllocator {
    RegisterNCCLSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<NCCLSymmetricMemoryAllocator>();
    // Query backend used for CUDA tensor
    if (getSymmMemBackendCUDA() == "NCCL") {
      // Direct set (static registration)
      register_allocator(
          c10::DeviceType::CUDA,
          allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("NCCL", allocator);
    }
  }
};

static RegisterNCCLSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
#endif // NCCL_HAS_SYMMEM_SUPPORT
