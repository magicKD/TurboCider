#include "../../../models/relayvsr/wan_weight_stream.h"
#include "../streaming_metadata.hpp"
#include "convrot.h"

#include <fcntl.h>
#include <sys/event.h>
#include <sys/stat.h>
#include <algorithm>
#include <span>

namespace rv {
namespace {
using tc::require;
namespace metadata = tc::streaming::metadata;

std::string wan_x2_name(const std::string& name) {
  using Pair = std::pair<std::string_view, std::string_view>;
  static constexpr Pair global[] = {
      {"positive_conditioning", "prompt_embeds"},
      {"time_embedding.0.", "condition_embedder.time_embedder.linear_1."},
      {"time_embedding.2.", "condition_embedder.time_embedder.linear_2."},
      {"time_projection.1.", "condition_embedder.time_proj."},
      {"text_embedding.0.", "condition_embedder.text_embedder.linear_1."},
      {"text_embedding.2.", "condition_embedder.text_embedder.linear_2."},
      {"lq_proj_in.linear.", "lq_proj.out_proj."},
      {"lq_proj_in.", "lq_proj."},
      {"head.modulation", "scale_shift_table"},
      {"head.head.", "proj_out."},
  };
  static constexpr Pair block[] = {
      {"modulation", "scale_shift_table"},
      {"self_attn.q.", "attn1.to_q."},
      {"self_attn.k.", "attn1.to_k."},
      {"self_attn.v.", "attn1.to_v."},
      {"self_attn.o.", "attn1.to_out.0."},
      {"self_attn.norm_", "attn1.norm_"},
      {"cross_attn.q.", "attn2.to_q."},
      {"cross_attn.k.", "attn2.to_k."},
      {"cross_attn.v.", "attn2.to_v."},
      {"cross_attn.o.", "attn2.to_out.0."},
      {"cross_attn.norm_", "attn2.norm_"},
      {"norm3.", "norm2."},
      {"ffn.0.", "ffn.net.0.proj."},
      {"ffn.2.", "ffn.net.2."},
  };
  const auto dot = name.starts_with("blocks.") ? name.find('.', 7) : std::string::npos;
  const auto start = dot == std::string::npos ? 0 : dot + 1;
  std::string_view suffix(name);
  suffix.remove_prefix(start);
  for (auto [key, replacement] : start ? std::span<const Pair>(block) : std::span<const Pair>(global)) {
    if (suffix.starts_with(key)) {
      auto result = name.substr(0, start);
      result.append(replacement);
      result.append(suffix.substr(key.size()));
      return result;
    }
  }
  return name;
}

at::ScalarType weight_type(NSString* name) {
  if ([name isEqual:@"I8"]) return at::kChar;
  if ([name isEqual:@"BF16"]) return at::kBFloat16;
  if ([name isEqual:@"F16"]) return at::kHalf;
  if ([name isEqual:@"F32"]) return at::kFloat;
  if ([name isEqual:@"U8"]) return at::kByte;
  throw std::runtime_error("Unsupported RelayVSR safetensors dtype");
}

WeightSlice weight_slice(id raw, uint64_t offset, uint64_t size) {
  require([raw isKindOfClass:NSDictionary.class], "Invalid RelayVSR tensor entry");
  NSDictionary* entry = raw;
  NSArray* dimensions = entry[@"shape"];
  NSArray* offsets = entry[@"data_offsets"];
  require([dimensions isKindOfClass:NSArray.class] && dimensions.count <= 5 &&
          [offsets isKindOfClass:NSArray.class] && offsets.count == 2 &&
          [entry[@"dtype"] isKindOfClass:NSString.class], "Invalid RelayVSR tensor shape/dtype");
  const auto begin = metadata::json_unsigned_integer(offsets[0], "tensor start", require);
  const auto end = metadata::json_unsigned_integer(offsets[1], "tensor end", require);
  require(begin <= end && end <= size - offset, "Invalid RelayVSR tensor offsets");
  WeightSlice slice{weight_type(entry[@"dtype"]), {}, offset + begin, end - begin};
  uint64_t bytes = c10::elementSize(slice.type);
  for (id dim in dimensions) {
    const auto length = metadata::json_unsigned_integer(dim, "tensor dimension", require);
    require(!length || bytes <= size / length, "RelayVSR tensor size overflow");
    bytes *= length;
    slice.shape.push_back(static_cast<int64_t>(length));
  }
  require(slice.bytes == bytes, "RelayVSR tensor length mismatch");
  return slice;
}

WeightIndex load_index(int fd, uint64_t size, RelayCheckpoint checkpoint) {
  const auto [header, start] = metadata::read_safetensors_header(
      fd, size, 8ull << 20, "Incomplete RelayVSR checkpoint", "Invalid RelayVSR checkpoint", require);
  NSDictionary* attributes = header[@"__metadata__"];
  require([attributes isKindOfClass:NSDictionary.class] &&
          [attributes[@"config"] isKindOfClass:NSString.class], "Missing RelayVSR checkpoint metadata");
  const bool wan = checkpoint == RelayCheckpoint::wan;
  const auto expected = wan ? @"relayvsr-wan-x2-v1" : @"relayvsr-flash-x2-v1";
  auto config = metadata::json_dictionary([attributes[@"config"] dataUsingEncoding:NSUTF8StringEncoding],
                                          "Invalid RelayVSR 2× configuration", require);
  require([attributes[@"format"] isEqual:expected] && [config[@"upscale_factor"] isEqual:@2],
          "Incompatible RelayVSR 2× checkpoint");

  WeightIndex index;
  for (NSString* key in header) {
    if ([key isEqualToString:@"__metadata__"]) continue;
    auto name = std::string(key.UTF8String);
    auto mapped = wan ? wan_x2_name(name) : std::move(name);
    auto slice = weight_slice(header[key], start, size);
    require(index.emplace(std::move(mapped), std::move(slice)).second, "Duplicate RelayVSR tensor");
  }
  if (!wan) return index;

  // ConvRot metadata is small and stored as a U8 tensor by Comfy. Verify
  // the exact matrix packing used by the Metal and MPS kernels up front.
  int quantized = 0;
  for (const auto& [name, field] : index) {
    constexpr std::string_view suffix = ".comfy_quant";
    if (!name.ends_with(suffix)) continue;
    require(field.type == at::kByte && field.bytes > 0 && field.bytes <= 512,
            "Invalid RelayVSR ConvRot tag");
    auto settings = metadata::read_json_dictionary(fd, field.bytes, field.offset,
        "Incomplete RelayVSR ConvRot tag", "Invalid RelayVSR ConvRot tag", require);
    const auto prefix = name.substr(0, name.size() - suffix.size());
    const auto& matrix = index.at(prefix + ".weight");
    const auto& scale = index.at(prefix + ".weight_scale");
    const auto& bias = index.at(prefix + ".bias");
    const auto& dims = matrix.shape;
    require([settings[@"format"] isEqual:@"int8_tensorwise"] &&
            [settings[@"convrot"] isEqual:@YES] && [settings[@"convrot_groupsize"] isEqual:@256] &&
            matrix.type == at::kChar && dims.size() == 2 && dims[0] > 0 && dims[0] % 128 == 0 &&
            dims[1] > 0 && dims[1] % 512 == 0 &&
            scale.type == at::kFloat && scale.shape == std::vector<int64_t>{dims[0], 1} &&
            bias.type == at::kBFloat16 && bias.shape == std::vector<int64_t>{dims[0]},
            "Incompatible RelayVSR ConvRot tensor");
    ++quantized;
  }
  require(quantized == 300, "Incomplete RelayVSR ConvRot weights");
  return index;
}
}  // namespace

struct Weights::File {
  int fd = -1, changes = -1;
  struct stat opened {};
  bool watch_registered = false;

  ~File() {
    if (changes >= 0) close(changes);
    if (fd >= 0) close(fd);
  }
};

Weights::Weights(const std::filesystem::path& path, RelayCheckpoint checkpoint, bool prefer_tensorops)
    : file_(std::make_unique<File>()), checkpoint_(checkpoint),
      tensorops_(checkpoint == RelayCheckpoint::wan && prefer_tensorops && tensorops_int8_available()) {
  @autoreleasepool {
    auto& file = *file_;
    file.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    require(file.fd >= 0, "Cannot open RelayVSR checkpoint: " + path.string());
    require(fstat(file.fd, &file.opened) == 0 && S_ISREG(file.opened.st_mode) && file.opened.st_size >= 8,
            "Invalid RelayVSR checkpoint: " + path.string());
    // Hold the original file descriptor throughout inference. Record vnode
    // writes as well as timestamps so a replacement link can safely coexist
    // with a running job on the original, unchanged checkpoint.
    file.changes = kqueue();
    if (file.changes >= 0) {
      struct kevent event;
      EV_SET(&event, file.fd, EVFILT_VNODE, EV_ADD | EV_CLEAR,
             NOTE_WRITE | NOTE_EXTEND | NOTE_REVOKE, 0, nullptr);
      file.watch_registered = kevent(file.changes, &event, 1, nullptr, 0, nullptr) == 0;
    }
    index_ = load_index(file.fd, file.opened.st_size, checkpoint);
    verify_source();
  }
}

Weights::~Weights() = default;

void Weights::verify_source() const {
  const auto& file = *file_;
  struct stat current {};
  const auto equal = [](timespec a, timespec b) { return a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec; };
  bool intact = fstat(file.fd, &current) == 0 && current.st_size == file.opened.st_size &&
                equal(current.st_mtimespec, file.opened.st_mtimespec);
  if (intact && file.watch_registered) {
    struct kevent event;
    const timespec immediate{};
    int count;
    do { count = kevent(file.changes, nullptr, 0, &event, 1, &immediate); }
    while (count < 0 && errno == EINTR);
    intact = count == 0;
  }
  const bool unlinked = current.st_nlink < file.opened.st_nlink && file.watch_registered;
  require(intact && (equal(current.st_ctimespec, file.opened.st_ctimespec) || unlinked),
          "RelayVSR checkpoint changed during inference");
}

Tensor load_cpu_weight(int fd, const WeightSlice& slice, const std::function<void()>& checkpoint,
                       bool background) {
  Tensor tensor;
  try {
    tensor = at::empty(slice.shape, at::TensorOptions().device(at::kCPU).dtype(slice.type));
  } catch (const c10::Error&) {
    // LibTorch's CPU allocator can report allocation failure as c10::Error.
    // Only this allocation retries through the single-slot path.
    if (background) throw std::bad_alloc{};
    throw;
  }
  auto* data = static_cast<char*>(tensor.data_ptr());
  for (uint64_t position = 0; position < slice.bytes;) {
    if (checkpoint) checkpoint();
    const auto length = std::min<uint64_t>(4ull << 20, slice.bytes - position);
    metadata::pread_exact(fd, data + position, length, slice.offset + position,
                          "Incomplete RelayVSR weight read", require);
    position += length;
  }
  return tensor;
}

Tensor to_gpu_weight(Tensor cpu, const std::string& name, bool tensorops) {
  const auto target = !cpu.is_floating_point() || (tensorops && name.ends_with(".weight_scale"))
      ? cpu.scalar_type() : at::kBFloat16;
  return cpu.to(at::kMPS, target);
}

void trim_mps_cache() {
  auto* allocator = at::mps::getIMPSAllocator();
  const auto active = allocator->getCurrentAllocatedMemory();
  const auto total = allocator->getDriverAllocatedMemory();
  if (total > active && total - active > (768ull << 20))
    allocator->emptyCache();
}

void Weights::enable_wan_streaming(std::atomic<bool>& cancelled) {
  require(convrot() && !stream_ && cache_.empty(), "Wan streaming requires fresh RelayVSR weights");
  stream_ = std::make_unique<WanWeightStream>(file_->fd, index_, cancelled, tensorops_);
}

void Weights::with_wan_block(int index, WanBlockUse use, const std::function<void()>& compute) {
  stream_->run(index, use, compute);
}

void Weights::release_wan_conditioner() {
  std::erase_if(cache_, [](const auto& item) {
    return item.first == "prompt_embeds" || item.first.starts_with("condition_embedder.");
  });
}

WanStreamStats Weights::wan_stream_stats() const { return stream_->stats(); }
bool Weights::contains(const std::string& name) const { return index_.contains(name); }

Tensor Weights::get(const std::string& name) const {
  if (auto cached = cache_.find(name); cached != cache_.end()) return cached->second;
  const auto& slice = index_.at(name);
  if (stream_ && name.starts_with("blocks.")) return stream_->get(name, slice);
  auto tensor = to_gpu_weight(load_cpu_weight(file_->fd, slice, {}), name, tensorops_);
  cache_.emplace(name, tensor);
  return tensor;
}

void Weights::record_int8(size_t projections, bool gelu) const {
  ++convrot_stats_.quantizations;
  convrot_stats_.projections += projections;
  convrot_stats_.fused_gelu += gelu ? projections : 0;
}
}  // namespace rv
