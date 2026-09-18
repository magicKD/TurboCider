#include "mlx_weight_pager.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace tc::streaming {
namespace {

using Clock = std::chrono::steady_clock;

void pager_require(bool value, const std::string &reason) {
    if (!value)
        throw std::invalid_argument("mlx_weight_pager: " + reason);
}

mx::Shape mlx_shape(const FieldSpec &field) {
    pager_require(field.materialization.has_value(),
                  "field is missing materialization");
    const auto &shape = field.materialization->shape;
    pager_require(!shape.empty() && shape.size() <= 8,
                  "field shape is invalid");
    uint64_t elements = 1;
    mx::Shape result;
    result.reserve(shape.size());
    for (const uint64_t dimension : shape) {
        pager_require(dimension && dimension <= INT32_MAX &&
                          dimension <= UINT64_MAX / elements,
                      "field shape overflows MLX");
        elements *= dimension;
        result.push_back(static_cast<int32_t>(dimension));
    }
    pager_require(elements <= UINT64_MAX / sizeof(uint16_t) &&
                      elements * sizeof(uint16_t) == field.bytes,
                  "field bytes differ from BF16 shape");
    return result;
}

const SourceRange &direct_source(const FieldSpec &field) {
    pager_require(field.materialization.has_value(),
                  "field is missing materialization");
    const auto &materialization = *field.materialization;
    pager_require(materialization.format == "BF16" &&
                      materialization.storage_mode == "mlx-metal-shared" &&
                      materialization.conversion == "copy-bf16-v1" &&
                      materialization.derived_from.empty() &&
                      materialization.derived_offset == 0 &&
                      materialization.reads.size() == 1,
                  "only direct BF16 MLX materialization is supported");
    const auto &source = materialization.reads.front();
    pager_require(source.dtype == "BF16" && source.bytes == field.bytes &&
                      source.shape == materialization.shape &&
                      !source.tensor.empty(),
                  "source range differs from field materialization");
    return source;
}

void check_cancel(const std::atomic<bool> *cancel) {
    if (cancel && cancel->load(std::memory_order_acquire))
        throw std::runtime_error("streaming_cancelled");
}

} // namespace

struct MlxWeightPager::State {
    struct Artifact {
        std::filesystem::path path;
        int descriptor = -1;
        uint64_t bytes = 0;
        dev_t device = 0;
        ino_t inode = 0;
        time_t modified_seconds = 0;
        long modified_nanos = 0;
    };
    struct Slot {
        std::vector<Tensor> arrays;
        std::vector<char *> pointers;
        uint32_t block = std::numeric_limits<uint32_t>::max();
    };
    struct Pool {
        std::string layout_class;
        std::vector<Slot> slots;
    };

    const Descriptor *descriptor = nullptr;
    const StageDescriptor *stage = nullptr;
    const StageLayout *layout = nullptr;
    std::shared_ptr<const SourceLease> lease;
    std::vector<Artifact> artifacts;
    std::map<uint32_t, const BlockSpec *> blocks;
    std::map<uint32_t, Pool> pools;
    Slot resident;
    bool resident_loaded = false;

    ~State() {
        for (auto &artifact : artifacts)
            if (artifact.descriptor >= 0)
                ::close(artifact.descriptor);
    }
};

MlxWeightPager::MlxWeightPager(
    const std::filesystem::path &artifact_root, const Descriptor &descriptor,
    const StageDescriptor &stage, const StageLayout &layout)
    : state_(std::make_unique<State>()) {
    pager_require(std::filesystem::is_directory(artifact_root),
                  "artifact root is missing");
    pager_require(stage.id == layout.id && !layout.resident &&
                      !layout.pools.empty() && !layout.groups.empty(),
                  "descriptor and layout stage differ");
    pager_require(descriptor.artifacts.size() <= max_stages,
                  "artifact count exceeds runtime limit");
    state_->descriptor = &descriptor;
    state_->stage = &stage;
    state_->layout = &layout;

    state_->artifacts.reserve(descriptor.artifacts.size());
    for (const auto &source : descriptor.artifacts) {
        const std::filesystem::path relative(source.id);
        pager_require(relative == relative.filename() &&
                          relative.extension() == ".safetensors",
                      "artifact id is not a safetensors filename");
        State::Artifact artifact;
        artifact.path = (artifact_root / relative).lexically_normal();
        artifact.descriptor = ::open(
            artifact.path.c_str(), O_RDONLY | O_CLOEXEC);
        pager_require(artifact.descriptor >= 0,
                      "cannot open artifact " + artifact.path.string());
        struct stat status{};
        pager_require(::fstat(artifact.descriptor, &status) == 0 &&
                          S_ISREG(status.st_mode) &&
                          static_cast<uint64_t>(status.st_size) == source.bytes,
                      "artifact size or type changed");
        artifact.bytes = source.bytes;
        artifact.device = status.st_dev;
        artifact.inode = status.st_ino;
        artifact.modified_seconds = status.st_mtimespec.tv_sec;
        artifact.modified_nanos = status.st_mtimespec.tv_nsec;
#ifdef F_NOCACHE
        pager_require(::fcntl(artifact.descriptor, F_NOCACHE, 1) == 0,
                      "cannot disable artifact file caching");
#endif
        state_->artifacts.push_back(std::move(artifact));
    }
    pager_require(!state_->artifacts.empty(), "descriptor has no artifacts");

    for (const auto &block : stage.blocks) {
        pager_require(state_->blocks.emplace(block.id, &block).second,
                      "duplicate block id");
        for (const auto &field : block.fields) {
            (void)mlx_shape(field);
            const auto &source = direct_source(field);
            pager_require(source.artifact < state_->artifacts.size(),
                          "block source artifact is unavailable");
            const auto &artifact = state_->artifacts[source.artifact];
            pager_require(source.offset <= artifact.bytes &&
                              source.bytes <= artifact.bytes - source.offset,
                          "block source range exceeds artifact");
        }
    }
    for (const auto &field : stage.resident_fields) {
        (void)mlx_shape(field);
        const auto &source = direct_source(field);
        pager_require(source.artifact < state_->artifacts.size(),
                      "resident source artifact is unavailable");
        const auto &artifact = state_->artifacts[source.artifact];
        pager_require(source.offset <= artifact.bytes &&
                          source.bytes <= artifact.bytes - source.offset,
                      "resident source range exceeds artifact");
    }
    for (const auto &group : layout.groups) {
        pager_require(group.blocks.size() == 1,
                      "pager currently requires one block per group");
        const auto found = state_->blocks.find(group.blocks.front());
        pager_require(found != state_->blocks.end(),
                      "layout group references an unknown block");
        pager_require(found->second->fields.size() == group.field_bytes.size(),
                      "layout group field count differs from descriptor");
        uint64_t group_bytes = 0;
        for (size_t index = 0; index < found->second->fields.size(); ++index) {
            const auto &field = found->second->fields[index];
            pager_require(field.alignment &&
                              (field.alignment & (field.alignment - 1)) == 0,
                          "descriptor field alignment is invalid");
            const uint64_t padding = field.alignment - 1;
            pager_require(field.bytes <= UINT64_MAX - padding,
                          "descriptor aligned field size overflows");
            const uint64_t aligned_bytes =
                (field.bytes + padding) & ~padding;
            pager_require(aligned_bytes == group.field_bytes[index],
                          "layout group field geometry differs from descriptor");
            pager_require(field.bytes <= UINT64_MAX - group_bytes,
                          "layout group byte count overflows");
            group_bytes += field.bytes;
        }
        pager_require(group.bytes == group_bytes,
                      "layout group byte count differs from descriptor");
    }
    check_open_files();
}

MlxWeightPager::MlxWeightPager(
    std::shared_ptr<const SourceLease> lease,
    const Descriptor &descriptor, const StageDescriptor &stage,
    const StageLayout &layout)
    : state_(std::make_unique<State>()) {
    pager_require(lease != nullptr, "source lease is unavailable");
    pager_require(stage.id == layout.id && !layout.resident &&
                      !layout.pools.empty() && !layout.groups.empty(),
                  "descriptor and layout stage differ");
    pager_require(descriptor.artifacts.size() <= max_stages,
                  "artifact count exceeds runtime limit");
    state_->descriptor = &descriptor;
    state_->stage = &stage;
    state_->layout = &layout;
    state_->lease = std::move(lease);

    state_->artifacts.reserve(descriptor.artifacts.size());
    for (const auto &source : descriptor.artifacts) {
        const std::filesystem::path id(source.id);
        pager_require(id == id.filename() && id.extension() == ".safetensors",
                      "artifact id is not a safetensors filename");
        State::Artifact artifact;
        artifact.path = state_->lease->file(source.id).path;
        artifact.descriptor =
            state_->lease->duplicate_fd(source.id).release();
        struct stat status{};
        pager_require(::fstat(artifact.descriptor, &status) == 0 &&
                          S_ISREG(status.st_mode) &&
                          static_cast<uint64_t>(status.st_size) == source.bytes,
                      "leased artifact size or type changed");
        artifact.bytes = source.bytes;
        artifact.device = status.st_dev;
        artifact.inode = status.st_ino;
        artifact.modified_seconds = status.st_mtimespec.tv_sec;
        artifact.modified_nanos = status.st_mtimespec.tv_nsec;
#ifdef F_NOCACHE
        pager_require(::fcntl(artifact.descriptor, F_NOCACHE, 1) == 0,
                      "cannot disable leased artifact file caching");
#endif
        state_->artifacts.push_back(std::move(artifact));
    }
    pager_require(!state_->artifacts.empty(), "descriptor has no artifacts");
    for (const auto &block : stage.blocks) {
        pager_require(state_->blocks.emplace(block.id, &block).second,
                      "duplicate block id");
        for (const auto &field : block.fields) {
            (void)mlx_shape(field);
            const auto &source = direct_source(field);
            pager_require(source.artifact < state_->artifacts.size(),
                          "block source artifact is unavailable");
            const auto &artifact = state_->artifacts[source.artifact];
            pager_require(source.offset <= artifact.bytes &&
                              source.bytes <= artifact.bytes - source.offset,
                          "block source range exceeds artifact");
        }
    }
    for (const auto &field : stage.resident_fields) {
        (void)mlx_shape(field);
        const auto &source = direct_source(field);
        pager_require(source.artifact < state_->artifacts.size(),
                      "resident source artifact is unavailable");
        const auto &artifact = state_->artifacts[source.artifact];
        pager_require(source.offset <= artifact.bytes &&
                          source.bytes <= artifact.bytes - source.offset,
                      "resident source range exceeds artifact");
    }
    for (const auto &group : layout.groups) {
        pager_require(group.blocks.size() == 1,
                      "pager currently requires one block per group");
        const auto found = state_->blocks.find(group.blocks.front());
        pager_require(found != state_->blocks.end(),
                      "layout group references an unknown block");
        pager_require(found->second->fields.size() == group.field_bytes.size(),
                      "layout group field count differs from descriptor");
        uint64_t group_bytes = 0;
        for (size_t index = 0; index < found->second->fields.size(); ++index) {
            const auto &field = found->second->fields[index];
            pager_require(field.alignment &&
                              (field.alignment & (field.alignment - 1)) == 0,
                          "descriptor field alignment is invalid");
            const uint64_t padding = field.alignment - 1;
            pager_require(field.bytes <= UINT64_MAX - padding,
                          "descriptor aligned field size overflows");
            const uint64_t aligned_bytes =
                (field.bytes + padding) & ~padding;
            pager_require(aligned_bytes == group.field_bytes[index],
                          "layout group field geometry differs from descriptor");
            pager_require(field.bytes <= UINT64_MAX - group_bytes,
                          "layout group byte count overflows");
            group_bytes += field.bytes;
        }
        pager_require(group.bytes == group_bytes,
                      "layout group byte count differs from descriptor");
    }
    check_open_files();
}

MlxWeightPager::~MlxWeightPager() = default;

void MlxWeightPager::check_open_files() const {
    pager_require(state_ != nullptr, "pager state is unavailable");
    if (state_->lease) {
        state_->lease->revalidate_open_files();
        state_->lease->revalidate_paths();
        return;
    }
    for (const auto &artifact : state_->artifacts) {
        struct stat opened{};
        struct stat named{};
        pager_require(::fstat(artifact.descriptor, &opened) == 0 &&
                          ::stat(artifact.path.c_str(), &named) == 0 &&
                          S_ISREG(opened.st_mode) && S_ISREG(named.st_mode) &&
                          opened.st_dev == artifact.device &&
                          opened.st_ino == artifact.inode &&
                          named.st_dev == artifact.device &&
                          named.st_ino == artifact.inode &&
                          static_cast<uint64_t>(opened.st_size) == artifact.bytes &&
                          static_cast<uint64_t>(named.st_size) == artifact.bytes &&
                          opened.st_mtimespec.tv_sec == artifact.modified_seconds &&
                          opened.st_mtimespec.tv_nsec == artifact.modified_nanos &&
                          named.st_mtimespec.tv_sec == artifact.modified_seconds &&
                          named.st_mtimespec.tv_nsec == artifact.modified_nanos,
                      "checkpoint_changed: opened artifact snapshot is stale");
    }
}

namespace {

void allocate_fields(MlxWeightPager::State::Slot &slot,
                     const std::vector<FieldSpec> &fields,
                     MlxWeightPagerMetrics &metrics) {
    pager_require(slot.arrays.empty() && slot.pointers.empty(),
                  "slot backing already exists");
    slot.arrays.reserve(fields.size());
    slot.pointers.reserve(fields.size());
    for (const auto &field : fields) {
        slot.arrays.emplace_back(mx::allocator::malloc(field.bytes),
                                 mlx_shape(field), mx::bfloat16);
        slot.pointers.push_back(slot.arrays.back().data<char>());
        ++metrics.slot_arrays_allocated;
    }
}

uint64_t read_fields(const std::vector<FieldSpec> &fields,
                     const std::vector<char *> &destinations,
                     const std::vector<MlxWeightPager::State::Artifact> &artifacts,
                     const std::atomic<bool> *cancel) {
    pager_require(fields.size() == destinations.size(),
                  "field destination count differs");
    uint64_t total = 0;
    for (size_t index = 0; index < fields.size(); ++index) {
        check_cancel(cancel);
        const auto &source = direct_source(fields[index]);
        pager_require(source.artifact < artifacts.size(),
                      "source artifact index is invalid");
        const auto &artifact = artifacts[source.artifact];
        pager_require(source.offset <= artifact.bytes &&
                          source.bytes <= artifact.bytes - source.offset,
                      "source range exceeds artifact");
        uint64_t done = 0;
        while (done < source.bytes) {
            check_cancel(cancel);
            const ssize_t count = ::pread(
                artifact.descriptor, destinations[index] + done,
                static_cast<size_t>(source.bytes - done),
                static_cast<off_t>(source.offset + done));
            if (count < 0 && errno == EINTR)
                continue;
            if (count <= 0)
                throw std::runtime_error(
                    "mlx_weight_pager: artifact read failed or was truncated");
            done += static_cast<uint64_t>(count);
        }
        pager_require(done <= UINT64_MAX - total,
                      "read byte counter overflow");
        total += done;
    }
    return total;
}

std::vector<std::string> binding_keys(
    const std::vector<FieldSpec> &fields) {
    std::vector<std::string> result;
    result.reserve(fields.size());
    for (const auto &field : fields)
        result.push_back(direct_source(field).tensor);
    return result;
}

} // namespace

void MlxWeightPager::load_resident(
    Weights &destination, const std::atomic<bool> *cancel) {
    pager_require(state_ && !state_->resident_loaded,
                  "resident weights are already loaded");
    check_open_files();
    const auto &fields = state_->stage->resident_fields;
    pager_require(!fields.empty(), "stage has no resident fields");
    allocate_fields(state_->resident, fields, metrics_);
    const auto begin = Clock::now();
    const uint64_t bytes = read_fields(
        fields, state_->resident.pointers, state_->artifacts, cancel);
    metrics_.resident_load_seconds +=
        std::chrono::duration<double>(Clock::now() - begin).count();
    metrics_.resident_bytes_loaded += bytes;
    pager_require(state_->layout->resident_source_read_bytes.has_value() &&
                      bytes == *state_->layout->resident_source_read_bytes,
                  "resident read count differs from compiled layout");
    destination.bind_arrays(binding_keys(fields), state_->resident.arrays);
    state_->resident_loaded = true;
}

void MlxWeightPager::create_pool(const PoolLayout &layout) {
    pager_require(state_ && !state_->pools.count(layout.id),
                  "pool backing already exists");
    const Group *representative_group = nullptr;
    for (const auto &group : state_->layout->groups)
        if (group.pool == layout.id) {
            representative_group = &group;
            break;
        }
    pager_require(representative_group &&
                      representative_group->blocks.size() == 1,
                  "pool has no one-block representative");
    const auto found = state_->blocks.find(
        representative_group->blocks.front());
    pager_require(found != state_->blocks.end() &&
                      found->second->layout_class == layout.layout_class,
                  "pool layout class differs from descriptor");
    const auto &fields = found->second->fields;
    State::Pool pool;
    pool.layout_class = layout.layout_class;
    pool.slots.resize(layout.slots.size());
    for (size_t slot_index = 0; slot_index < layout.slots.size(); ++slot_index) {
        const auto &slot_layout = layout.slots[slot_index];
        pager_require(slot_layout.field_capacity.size() == fields.size(),
                      "slot field count differs from descriptor");
        for (size_t field_index = 0; field_index < fields.size(); ++field_index)
            pager_require(fields[field_index].bytes <=
                              slot_layout.field_capacity[field_index],
                          "slot field capacity is too small");
        allocate_fields(pool.slots[slot_index], fields, metrics_);
    }
    state_->pools.emplace(layout.id, std::move(pool));
}

void MlxWeightPager::destroy_pool(uint32_t pool) noexcept {
    if (!state_) return;
    state_->pools.erase(pool);
}

uint64_t MlxWeightPager::fill(
    const Group &group, const tc_stream_slot_ticket_v1 &ticket,
    const std::atomic<bool> *cancel) {
    pager_require(state_ && group.blocks.size() == 1 &&
                      ticket.pool == group.pool && ticket.slot == group.slot,
                  "fill identity differs from group");
    auto pool = state_->pools.find(group.pool);
    pager_require(pool != state_->pools.end() &&
                      ticket.slot < pool->second.slots.size(),
                  "fill pool or slot is unavailable");
    const auto block = state_->blocks.find(group.blocks.front());
    pager_require(block != state_->blocks.end() &&
                      block->second->layout_class == pool->second.layout_class,
                  "fill block class differs from pool");
    auto &slot = pool->second.slots[ticket.slot];
    const auto begin = Clock::now();
    const uint64_t bytes = read_fields(
        block->second->fields, slot.pointers, state_->artifacts, cancel);
    const double seconds =
        std::chrono::duration<double>(Clock::now() - begin).count();
    pager_require(bytes == group.bytes,
                  "fill byte count differs from compiled group");
    slot.block = group.blocks.front();
    {
        std::lock_guard lock(metrics_mutex_);
        metrics_.streamed_bytes_loaded += bytes;
        metrics_.streamed_load_seconds += seconds;
        ++metrics_.slot_fills;
        if (seconds > metrics_.maximum_fill_seconds) {
            metrics_.maximum_fill_seconds = seconds;
            metrics_.maximum_fill_group = group.id;
        }
    }
    return bytes;
}

Weights MlxWeightPager::bind(
    const Group &group, const tc_stream_slot_ticket_v1 &ticket) const {
    pager_require(state_ && group.blocks.size() == 1 &&
                      ticket.pool == group.pool && ticket.slot == group.slot,
                  "bind identity differs from group");
    const auto pool = state_->pools.find(group.pool);
    pager_require(pool != state_->pools.end() &&
                      ticket.slot < pool->second.slots.size(),
                  "bind pool or slot is unavailable");
    const auto block = state_->blocks.find(group.blocks.front());
    pager_require(block != state_->blocks.end(),
                  "bind block is unavailable");
    const auto &slot = pool->second.slots[ticket.slot];
    pager_require(slot.block == group.blocks.front(),
                  "slot content identity differs from group");
    Weights result;
    result.bind_arrays(binding_keys(block->second->fields), slot.arrays);
    return result;
}

} // namespace tc::streaming
