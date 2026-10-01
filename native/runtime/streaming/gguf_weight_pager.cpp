#include "gguf_weight_pager.hpp"
#include "../../core/gguf_affine.hpp"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <set>
#include <thread>

namespace tc::streaming {
namespace {
using Clock = std::chrono::steady_clock;
void insist(bool valid, const char *reason) {
    if (!valid) throw std::invalid_argument(std::string("gguf_weight_pager: ") + reason);
}
void cancelled(const std::atomic<bool> *cancel) {
    if (cancel && cancel->load(std::memory_order_acquire)) throw Cancelled();
}
mx::Dtype mlx_dtype(std::string_view format) {
    if (format == "U32") return mx::uint32;
    if (format == "U8") return mx::uint8;
    if (format == "BF16") return mx::bfloat16;
    if (format == "F16") return mx::float16;
    if (format == "F32") return mx::float32;
    throw std::invalid_argument("gguf_weight_pager: unsupported target dtype");
}
gguf::DecodeDType decode_dtype(mx::Dtype dtype) {
    if (dtype == mx::bfloat16) return gguf::DecodeDType::bf16;
    if (dtype == mx::float16) return gguf::DecodeDType::f16;
    insist(dtype == mx::float32, "invalid target dtype");
    return gguf::DecodeDType::f32;
}
mx::Shape mlx_shape(const std::vector<uint64_t> &shape) {
    mx::Shape output;
    for (uint64_t dim : shape) {
        insist(dim && dim <= INT32_MAX, "shape dimension exceeds MLX");
        output.push_back(int32_t(dim));
    }
    insist(!output.empty(), "empty target shape");
    return output;
}
uint64_t aligned(uint64_t bytes) {
    return gguf::checked_add(bytes, GgufWeightPager::buffer_alignment - 1) &
           ~(GgufWeightPager::buffer_alignment - 1);
}
struct Backing {
    mx::allocator::Buffer buffer{nullptr};
    StorageLease lease;
    ~Backing() { if (buffer.ptr()) mx::allocator::free(buffer); }
};
Tensor allocate(MemoryLedger &ledger, uint64_t bytes, uint64_t upper,
                const mx::Shape &shape, mx::Dtype dtype, MemoryClass kind, uint64_t generation) {
    insist(bytes && upper >= bytes && upper % GgufWeightPager::buffer_alignment == 0,
            "invalid backing capacity");
    auto reservation = ledger.try_reserve(kind, upper, "gguf-resident-or-slot-v1");
    insist(reservation.has_value(), "managed backing budget insufficient");
    auto owner = std::make_shared<Backing>();
    owner->buffer = mx::allocator::malloc(size_t(bytes));
    insist(owner->buffer.ptr() != nullptr, "MLX allocation failed");
    const auto actual = mx::allocator::allocator().size(owner->buffer);
    insist(actual >= bytes && actual <= upper, "allocator capacity exceeds compiled upper");
    owner->lease = reservation->commit({0x544347475546ull,
        uint64_t(reinterpret_cast<uintptr_t>(owner->buffer.ptr())), uint64_t(actual), generation});
    // Data, not the pager, owns the physical allocation and ledger claim.
    return Tensor(owner->buffer, shape, dtype, [owner](mx::allocator::Buffer) {});
}
bool same_ticket(const tc_stream_slot_ticket_v1 &a, const tc_stream_slot_ticket_v1 &b) {
    return a.struct_size == b.struct_size && a.version == b.version && a.pool == b.pool && a.slot == b.slot &&
        a.request_generation == b.request_generation && a.content_generation == b.content_generation &&
        a.item.stage == b.item.stage && a.item.pass == b.item.pass &&
        a.item.step == b.item.step && a.item.group == b.item.group;
}
using Key = std::pair<uint32_t, std::string>;
struct RawTensor {
    gguf::TensorDescriptor descriptor;
    std::optional<Tensor> storage;
    const std::byte *pointer = nullptr;
    uint32_t cached_type = 0;
};
struct FieldRecipe {
    const FieldSpec *field = nullptr;
    const RawTensor *source = nullptr;
    mx::Dtype dtype = mx::bfloat16;
    std::optional<gguf::AffinePart> affine;
};
std::string binding_key(const FieldRecipe &recipe) {
    auto key=recipe.field->materialization->reads.front().tensor;
    if(recipe.affine && *recipe.affine!=gguf::AffinePart::codes) {
        insist(key.ends_with(".weight"), "affine tensor missing weight suffix");
        key.resize(key.size()-7);
        key+=*recipe.affine==gguf::AffinePart::scales ? ".scales" : ".biases";
    }
    return key;
}
struct Slot {
    std::vector<Tensor> arrays;
    std::vector<std::byte *> pointers;
    std::optional<tc_stream_slot_ticket_v1> content;
    uint32_t block = UINT32_MAX;
};
struct Pool { std::string layout_class; std::vector<Slot> slots; uint64_t upper = 0; };
}

struct GgufWeightPager::State {
    std::shared_ptr<const SourceLease> lease;
    const Descriptor &descriptor;
    const StageDescriptor &stage;
    const StageLayout &layout;
    MemoryLedger &ledger;
    std::thread::id owner = std::this_thread::get_id();
    std::vector<OwnedSourceFd> fds;
    std::vector<gguf::Directory> directories;
    std::map<Key, RawTensor> sources;
    std::map<uint32_t, std::vector<FieldRecipe>> recipes;
    std::vector<FieldRecipe> resident;
    std::map<uint32_t, Pool> pools;
    mutable std::mutex metrics_mutex;
    GgufWeightPagerMetrics metrics;
    bool loaded = false, failed = false, resident_bound = false;
    bool legacy_float = false;

    State(std::shared_ptr<const SourceLease> source, const Descriptor &d, const StageDescriptor &s,
          const StageLayout &l, MemoryLedger &m)
        : lease(std::move(source)), descriptor(d), stage(s), layout(l), ledger(m) {}
    void check_owner() const { insist(owner == std::this_thread::get_id(), "owner thread mismatch"); }
    FieldRecipe recipe(const FieldSpec &field, bool alias) {
        insist(field.materialization.has_value(), "field missing materialization");
        const auto &m = *field.materialization;
        const bool packed_source = alias && m.conversion == "gguf-packed-gather-source-v1";
        std::optional<gguf::AffinePart> affine;
        if (m.conversion=="gguf-affine-codes-v1") affine=gguf::AffinePart::codes;
        if (m.conversion=="gguf-affine-scales-v1") affine=gguf::AffinePart::scales;
        if (m.conversion=="gguf-affine-biases-v1") affine=gguf::AffinePart::biases;
        insist(m.storage_mode == "mlx-metal-shared" && m.reads.size() == 1 && m.derived_from.empty() &&
                   m.derived_offset == 0 && (m.conversion == (alias ? "gguf-native-alias-v1" : "gguf-cpu-rne-v1") ||
                     (alias && legacy_float && m.conversion=="gguf-mlx-float-alias-v1") ||
                     packed_source || (affine && !alias)),
                "unsupported typed materialization");
        const auto &r = m.reads.front();
        insist(r.artifact < directories.size(), "source artifact out of bounds");
        const auto &tensor = directories[r.artifact].tensor(r.tensor);
        insist(tensor.file_offset == r.offset && tensor.bytes == r.bytes && tensor.logical_shape() == r.shape &&
                   r.dtype == gguf::type_info(tensor.type).name, "source recipe differs from held directory");
        const auto original = tensor.logical_shape();
        const bool pad_reshape = (tensor.name == "x_pad_token" || tensor.name == "cap_pad_token") &&
            original.size() == 1 && m.shape == std::vector<uint64_t>{1, original[0]};
        if (packed_source) {
            insist(m.format == "U8" && m.shape == std::vector<uint64_t>{tensor.bytes} &&
                   field.bytes == tensor.bytes && original.size() == 2 && !legacy_float,
                   "invalid packed gather source geometry/profile");
        } else if(affine) {
            insist(original.size()==2 && (tensor.type==2 || tensor.type==3 || tensor.type==8), "unsupported affine source");
            const uint32_t bits=tensor.type==8 ? 8 : 4;
            const std::vector<uint64_t> expected{tensor.rows(), *affine==gguf::AffinePart::codes ? tensor.columns()*bits/32 : tensor.columns()/32};
            insist(m.shape==expected && m.format==(*affine==gguf::AffinePart::codes ? "U32":"F16"), "affine target geometry mismatch");
        } else insist(m.shape == original || pad_reshape, "target shape differs from source");
        const auto dtype = mlx_dtype(m.format);
        uint64_t elements=1; for(uint64_t dim:m.shape) elements=gguf::checked_mul(elements,dim);
        const uint32_t item=dtype==mx::uint8 ? 1 : dtype==mx::uint32 ? 4 : gguf::dtype_bytes(decode_dtype(dtype));
        insist(field.bytes == gguf::checked_mul(elements, item),
                "target byte count differs from dtype/shape");
        insist(field.alignment == buffer_alignment, "unqualified backing alignment");
        if (alias && !packed_source) {
            insist((tensor.type == 0 && dtype == mx::float32) || (tensor.type == 1 && dtype == mx::float16) ||
                       (tensor.type == 30 && dtype == (legacy_float ? mx::float16 : mx::bfloat16)), "resident alias would require conversion");
        }
        const Key key{r.artifact, r.tensor};
        auto [it, fresh] = sources.try_emplace(key);
        if (fresh) { it->second.descriptor = tensor; it->second.cached_type=legacy_float && tensor.type==30 ? 1 : tensor.type; }
        return {&field, &it->second, dtype, affine};
    }
};

GgufWeightPager::GgufWeightPager(std::shared_ptr<const SourceLease> lease, const Descriptor &descriptor,
        const StageDescriptor &stage, const StageLayout &layout, MemoryLedger &ledger)
    : state_(std::make_unique<State>(std::move(lease), descriptor, stage, layout, ledger)) {
    auto &s = *state_;
    const auto float_loader=descriptor.workload.find("float_loader");
    s.legacy_float=float_loader!=descriptor.workload.end() && float_loader->second=="mlx-bf16-to-f16-v1";
    insist(s.lease && s.lease->has_verified_content() && stage.id == layout.id && !layout.resident && !layout.groups.empty(), "invalid/unverified source/layout");
    insist(descriptor.artifacts.size() && descriptor.artifacts.size() <= 64, "artifact count out of bounds");
    check_unchanged();
    for (const auto &artifact : descriptor.artifacts) {
        const auto &file = s.lease->file(artifact.id);
        insist(file.path.extension() == ".gguf" && file.bytes == artifact.bytes &&
            artifact.identity_kind == SourceIdentityKind::content_sha256 && artifact.identity == file.content_digest,
            "artifact identity mismatch");
        s.fds.push_back(s.lease->duplicate_fd(artifact.id));
#ifdef F_NOCACHE
        insist(::fcntl(s.fds.back().get(), F_NOCACHE, 1) == 0, "cannot disable source file caching");
#endif
        s.directories.push_back(gguf::read_directory(s.fds.back().get(), file.bytes));
    }
    if (s.directories.size() > 1) gguf::validate_split_set(s.directories);
    for (const auto &block : stage.blocks) {
        insist(!s.recipes.count(block.id), "duplicate block ID");
        auto &recipes = s.recipes[block.id];
        for (const auto &field : block.fields) recipes.push_back(s.recipe(field, false));
    }
    for (const auto &field : stage.resident_fields) s.resident.push_back(s.recipe(field, true));
    for (const auto &group : layout.groups) {
        insist(group.blocks.size() == 1 && s.recipes.count(group.blocks.front()), "invalid compiled group");
        const auto &recipes = s.recipes.at(group.blocks.front());
        insist(recipes.size() == group.field_bytes.size(), "compiled field count mismatch");
        uint64_t bytes = 0;
        for (size_t i = 0; i < recipes.size(); ++i) {
            insist(aligned(recipes[i].field->bytes) == group.field_bytes[i], "compiled field capacity mismatch");
            bytes = gguf::checked_add(bytes, recipes[i].field->bytes);
        }
        insist(bytes == group.bytes, "compiled content byte mismatch");
    }
    check_unchanged();
}
GgufWeightPager::~GgufWeightPager() = default;

void GgufWeightPager::load_packed(const std::atomic<bool> *cancel) {
    auto &s = *state_; s.check_owner();
    insist(!s.loaded && !s.failed, "source already loaded or failed");
    check_unchanged();
    const auto begin = Clock::now();
    try {
        for (auto &[key, raw] : s.sources) {
            cancelled(cancel);
            const auto &d = raw.descriptor;
            const bool floating = gguf::type_info(d.type).elements == 1;
            mx::Dtype dtype = mx::uint8;
            mx::Shape shape;
            if (floating) {
                dtype = d.type == 0 ? mx::float32 : raw.cached_type == 1 ? mx::float16 : mx::bfloat16;
                shape = mlx_shape(d.logical_shape());
            } else {
                const auto &type = gguf::type_info(d.type);
                shape = mlx_shape({d.rows(), d.columns() / type.elements * type.bytes});
            }
            raw.storage = allocate(s.ledger, d.bytes, aligned(d.bytes), shape, dtype, MemoryClass::Weights, s.lease->generation());
            auto *pointer = raw.storage->data<std::byte>();
            uint64_t done = 0;
            while (done != d.bytes) {
                cancelled(cancel);
                const uint64_t amount = std::min<uint64_t>(d.bytes - done, 1ull << 20);
                const ssize_t count = ::pread(s.fds[key.first].get(), pointer + done, size_t(amount), off_t(d.file_offset + done));
                if (count < 0 && errno == EINTR) continue;
                insist(count > 0, "packed read failed or truncated"); done += uint64_t(count);
            }
            raw.pointer = pointer;
            if(d.type==30 && raw.cached_type==1) {
                // Explicit importer-compatible representation, converted in
                // place once; never retain a second full floating checkpoint.
                for(uint64_t i=0;i<d.elements;++i) {
                    if(!(i%4096))cancelled(cancel);
                    uint16_t bits;std::memcpy(&bits,pointer+i*2,2);
                    const auto value=std::bit_cast<float>(uint32_t(bits)<<16);
                    const auto half=gguf::float_to_fp16_rne(value);
                    std::memcpy(pointer+i*2,&half,2);
                }
            }
            // Validate float aliases without constructing a second dense copy.
            // decode_cpu_into forbids overlap, so use the static finite-value
            // classification here; normal refills use the shared decoder.
            if (floating) {
                for (uint64_t i = 0; i < d.elements; ++i) {
                    if (!(i % 4096)) cancelled(cancel);
                    uint32_t bits = 0;
                    const uint32_t width = d.type == 0 ? 4 : 2;
                    std::memcpy(&bits, pointer + i * width, width);
                    const uint32_t mask = d.type == 0 ? 0x7f800000 : raw.cached_type == 1 ? 0x7c00 : 0x7f80;
                    insist((bits & mask) != mask, "nonfinite source float alias");
                }
                s.metrics.source_float_bytes = gguf::checked_add(s.metrics.source_float_bytes, d.bytes);
            } else s.metrics.source_quantized_bytes = gguf::checked_add(s.metrics.source_quantized_bytes, d.bytes);
            s.metrics.packed_source_bytes = gguf::checked_add(s.metrics.packed_source_bytes, d.bytes);
            s.metrics.packed_capacity_bytes = gguf::checked_add(s.metrics.packed_capacity_bytes,
                mx::allocator::allocator().size(raw.storage->data_shared_ptr()->buffer));
        }
        cancelled(cancel); check_unchanged();
        s.loaded = true;
        s.metrics.packed_read_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    } catch (...) { s.failed = true; throw; }
}

void GgufWeightPager::load_resident_aliases(Weights &destination) {
    auto &s = *state_; s.check_owner(); insist(s.loaded && !s.failed && !s.resident_bound, "packed source unavailable/already bound");
    std::vector<std::string> keys; std::vector<Tensor> arrays;
    uint64_t alias_bytes = 0;
    for (const auto &recipe : s.resident) {
        if (recipe.field->materialization->conversion == "gguf-packed-gather-source-v1") continue;
        keys.push_back(binding_key(recipe));
        arrays.push_back(mx::reshape(*recipe.source->storage, mlx_shape(recipe.field->materialization->shape)));
        alias_bytes = gguf::checked_add(alias_bytes, recipe.field->bytes);
    }
    destination.bind_arrays(keys, arrays);
    s.metrics.fixed_alias_bytes = alias_bytes; s.resident_bound = true;
}

Tensor GgufWeightPager::gather_rows(const std::string &tensor, std::span<const uint64_t> rows,
                                  const std::atomic<bool> *cancel) {
    auto &s = *state_; s.check_owner();
    insist(s.loaded && !s.failed && !rows.empty() && rows.size() <= 1024, "invalid/unloaded embedding gather");
    const auto found = std::find_if(s.resident.begin(), s.resident.end(), [&](const FieldRecipe &r) {
        return r.source->descriptor.name == tensor && r.field->materialization->conversion == "gguf-packed-gather-source-v1";
    });
    insist(found != s.resident.end(), "embedding source not declared by descriptor");
    const auto &raw = *found->source; const auto &d = raw.descriptor;
    for (uint64_t row : rows) insist(row < d.rows(), "embedding token ID out of bounds");
    cancelled(cancel); check_unchanged();
    const auto dtype = raw.cached_type == 0 ? mx::float32 : raw.cached_type == 1 ? mx::float16 : mx::bfloat16;
    const auto bytes = gguf::checked_mul(gguf::checked_mul(rows.size(), d.columns()), gguf::dtype_bytes(decode_dtype(dtype)));
    auto output = allocate(s.ledger, bytes, aligned(bytes), mlx_shape({rows.size(), d.columns()}), dtype,
                           MemoryClass::Conditioning, s.lease->generation());
    gguf::decode_cpu_gather({{raw.pointer, size_t(d.bytes)}, raw.cached_type, d.rows(), d.columns()},
        rows, 0, d.columns(), {{output.data<std::byte>(), size_t(bytes)}, decode_dtype(dtype),
        d.columns() * gguf::dtype_bytes(decode_dtype(dtype)), gguf::dtype_bytes(decode_dtype(dtype))}, cancel);
    cancelled(cancel); check_unchanged();
    return output;
}

void GgufWeightPager::create_pool(const PoolLayout &layout) {
    auto &s = *state_; s.check_owner(); insist(s.loaded && !s.failed && !s.pools.count(layout.id), "invalid pool create");
    const auto group = std::find_if(s.layout.groups.begin(), s.layout.groups.end(),
        [&](const Group &g) { return g.pool == layout.id; });
    insist(group != s.layout.groups.end(), "pool lacks representative group");
    const auto &recipes = s.recipes.at(group->blocks.front());
    // Do not trust a forged caller layout to make incompatible shapes/dtypes
    // share a slot. Check every group before allocating any backing.
    for (const auto &g : s.layout.groups) {
        if (g.pool != layout.id) continue;
        const auto &other = s.recipes.at(g.blocks.front());
        insist(other.size() == recipes.size(), "pool has incompatible field counts");
        for (size_t i = 0; i < recipes.size(); ++i)
            insist(other[i].dtype == recipes[i].dtype &&
                other[i].field->materialization->shape == recipes[i].field->materialization->shape &&
                other[i].field->bytes == recipes[i].field->bytes, "pool has incompatible field layouts");
    }
    Pool pool; pool.layout_class = layout.layout_class;
    for (const auto &planned : layout.slots) {
        insist(planned.field_capacity.size() == recipes.size(), "slot field count mismatch");
        Slot slot;
        for (size_t i = 0; i < recipes.size(); ++i) {
            const auto &recipe = recipes[i];
            insist(planned.field_capacity[i] >= recipe.field->bytes, "slot field capacity too small");
            slot.arrays.push_back(allocate(s.ledger, recipe.field->bytes, planned.field_capacity[i],
                mlx_shape(recipe.field->materialization->shape), recipe.dtype, MemoryClass::RefillSlot, s.lease->generation()));
            slot.pointers.push_back(slot.arrays.back().data<std::byte>());
            pool.upper = gguf::checked_add(pool.upper, planned.field_capacity[i]);
        }
        pool.slots.push_back(std::move(slot));
    }
    s.metrics.dense_pool_capacity_bytes = gguf::checked_add(s.metrics.dense_pool_capacity_bytes, pool.upper);
    s.metrics.maximum_dense_pool_capacity_bytes = std::max(s.metrics.maximum_dense_pool_capacity_bytes, s.metrics.dense_pool_capacity_bytes);
    s.pools.emplace(layout.id, std::move(pool));
}
void GgufWeightPager::destroy_pool(uint32_t id) noexcept {
    auto &s = *state_;
    if (s.owner != std::this_thread::get_id()) std::terminate();
    const auto found = s.pools.find(id);
    if (found == s.pools.end()) return;
    s.metrics.dense_pool_capacity_bytes -= found->second.upper;
    s.pools.erase(found);
}
uint64_t GgufWeightPager::fill(const Group &group, const tc_stream_slot_ticket_v1 &ticket,
                              const std::atomic<bool> *cancel) {
    auto &s = *state_;
    insist(s.loaded && !s.failed && group.blocks.size() == 1 && ticket.pool == group.pool && ticket.slot == group.slot,
            "fill ticket mismatch");
    auto &pool = s.pools.at(group.pool); auto &slot = pool.slots.at(ticket.slot);
    slot.content.reset(); slot.block = UINT32_MAX;
    const auto &recipes = s.recipes.at(group.blocks.front());
    insist(recipes.size() == slot.arrays.size(), "fill field count mismatch");
    uint64_t output = 0, source_bytes = 0;
    const auto begin = Clock::now();
    for (size_t i = 0; i < recipes.size(); ++i) {
        const auto &recipe = recipes[i]; const auto &raw = *recipe.source;
        const auto &d = raw.descriptor;
        const gguf::PackedMatrix matrix{{raw.pointer, size_t(d.bytes)}, raw.cached_type, d.rows(), d.columns()};
        if(recipe.affine) {
            output=gguf::checked_add(output,gguf::pack_native_affine(matrix,*recipe.affine,
                {slot.pointers[i],size_t(recipe.field->bytes)},cancel));
            source_bytes=gguf::checked_add(source_bytes,d.bytes);
        } else {
            const auto receipt = gguf::decode_cpu_into(matrix, {0, d.rows(), 0, d.columns()},
                {{slot.pointers[i], size_t(recipe.field->bytes)}, decode_dtype(recipe.dtype),
                 d.columns() * gguf::dtype_bytes(decode_dtype(recipe.dtype)), gguf::dtype_bytes(decode_dtype(recipe.dtype))}, cancel);
            output = gguf::checked_add(output, receipt.bytes_written);
            source_bytes = gguf::checked_add(source_bytes, receipt.source_bytes_processed);
        }
    }
    cancelled(cancel); insist(output == group.bytes, "fill content byte count mismatch");
    slot.block = group.blocks.front(); slot.content = ticket;
    std::lock_guard lock(s.metrics_mutex);
    ++s.metrics.fill_count; s.metrics.decoded_bytes += output; s.metrics.source_bytes_processed += source_bytes;
    s.metrics.decode_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
    return output;
}
Weights GgufWeightPager::bind(const Group &group, const tc_stream_slot_ticket_v1 &ticket) const {
    auto &s = *state_; s.check_owner();
    insist(group.blocks.size() == 1 && ticket.pool == group.pool && ticket.slot == group.slot, "bind ticket mismatch");
    const auto &slot = s.pools.at(group.pool).slots.at(ticket.slot);
    insist(slot.content && same_ticket(*slot.content, ticket) && slot.block == group.blocks.front(), "stale/unready slot content");
    std::vector<std::string> keys;
    for (const auto &recipe : s.recipes.at(group.blocks.front())) keys.push_back(binding_key(recipe));
    Weights result; result.bind_arrays(keys, slot.arrays); return result;
}
void GgufWeightPager::check_unchanged() const {
    state_->lease->revalidate_open_files(); state_->lease->revalidate_paths();
}
GgufWeightPagerMetrics GgufWeightPager::metrics() const {
    state_->check_owner(); std::lock_guard lock(state_->metrics_mutex); return state_->metrics;
}
const gguf::Directory &GgufWeightPager::directory(uint32_t artifact) const { return state_->directories.at(artifact); }
} // namespace tc::streaming
