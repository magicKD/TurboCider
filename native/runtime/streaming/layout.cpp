#include "layout.hpp"
#include "../memory_manifest.hpp"

#include <algorithm>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tc::streaming {
namespace {
void check(bool ok, const std::string &why) {
    if (!ok) throw std::invalid_argument("streaming_layout_invalid: " + why);
}
uint64_t add(uint64_t a, uint64_t b) {
    check(b <= std::numeric_limits<uint64_t>::max() - a, "byte overflow");
    return a + b;
}
uint64_t aligned(uint64_t n, uint64_t a) {
    check(a && (a & (a - 1)) == 0, "alignment must be power of two");
    return add(n, a - 1) & ~(a - 1);
}
void field(std::ostream &o, const std::string &s) { o << s.size() << ':' << s << '|'; }
void identifier(const std::string &s) {
    check(!s.empty() && s.size() <= 4096 && s.find('\0') == std::string::npos,
          "invalid metadata identity");
}
void shape(std::ostream &out, const std::vector<uint64_t> &dims) {
    check(!dims.empty() && dims.size() <= 8, "invalid tensor/pass rank");
    uint64_t elements = 1;
    out << dims.size() << '[';
    for (auto dim : dims) {
        check(dim && dim <= UINT64_MAX / elements, "shape overflow/zero dimension");
        elements *= dim;
        out << dim << ',';
    }
    out << ']';
}
}

Layout compile_layout(const StreamingConfig &c, const Descriptor &d) {
    validate_streaming_config(c);
    check(c.active(), "streaming is disabled");
    identifier(d.model); identifier(d.checkpoint_identity); identifier(d.backend_revision);
    check(!d.stages.empty() && d.stages.size() <= max_stages, "stage count limit");
    Layout out;
    uint64_t identity_bytes = 0;
    auto identity = [&](const std::string &value) {
        identifier(value);
        identity_bytes = add(identity_bytes, value.size());
        check(identity_bytes <= max_identity_bytes, "plan_too_large: identity bytes");
    };
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "tc-stream-layout-v2|fixed-fields|request|";
    field(canonical, d.model); field(canonical, d.checkpoint_identity);
    field(canonical, d.backend_revision);
    check(d.artifacts.size() <= max_stages && d.workload.size() <= 256,
          "artifact/workload count limit");
    canonical << d.artifacts.size() << '|';
    std::set<std::string> artifact_ids;
    for (const auto &a : d.artifacts) {
        identity(a.id); identity(a.identity);
        check(a.bytes && artifact_ids.insert(a.id).second, "invalid/duplicate artifact");
        check(a.identity_kind == SourceIdentityKind::snapshot ||
              a.identity_kind == SourceIdentityKind::content_sha256, "invalid artifact identity kind");
        if (a.identity_kind == SourceIdentityKind::content_sha256)
            check(a.identity.size() == 64 && std::all_of(a.identity.begin(), a.identity.end(),
                [](unsigned char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); }),
                "invalid artifact SHA256");
        field(canonical, a.id); field(canonical, a.identity);
        canonical << a.bytes << ',' << static_cast<int>(a.identity_kind) << '|';
    }
    canonical << d.workload.size() << '|';
    for (const auto &[key, value] : d.workload) {
        identity(key); identity(value); field(canonical, key); field(canonical, value);
    }
    out.materializations_complete = !d.artifacts.empty() && !d.workload.empty();
    std::set<std::string> stages;
    uint64_t total_fields = 0, total_source_ranges = 0;
    for (const auto &sd : d.stages) {
        identity(sd.id); identity(sd.adapter_revision);
        check(stages.insert(sd.id).second, "duplicate stage " + sd.id);
        check(!sd.blocks.empty() && sd.blocks.size() <= max_blocks, "block count limit");
        check(sd.pass_count && sd.pass_count <= max_passes, "pass count limit");
        check(sd.passes.empty() || sd.passes.size() == sd.pass_count, "pass template count mismatch");
        out.materializations_complete &= !sd.passes.empty();
        check(sd.min_slots && sd.min_slots <= sd.max_slots && sd.max_slots <= max_slots,
              "invalid adapter slot range");
        check(sd.max_group_size > 0, "invalid adapter group range");
        const auto it = c.stages.find(sd.id);
        check(it != c.stages.end() || sd.fixed_policy.has_value(), "missing stage " + sd.id);
        const auto &sc = it != c.stages.end() ? it->second : *sd.fixed_policy;
        StreamingConfig single = c;
        single.stages = {{sd.id, sc}};
        validate_streaming_config(single);
        StageLayout stage;
        stage.id = sd.id; stage.pass_count = sd.pass_count;
        stage.inherited = it == c.stages.end();
        stage.resident = *sc.residency == "resident";
        stage.prefix = stage.resident ? static_cast<uint32_t>(sd.blocks.size()) : *sc.resident_prefix_blocks;
        stage.group_size = stage.resident ? 0 : *sc.block_group_size;
        stage.slot_count = stage.resident ? 0 : *sc.slot_count;
        stage.distance = stage.resident ? 0 : *sc.prefetch_distance;
        stage.workers = stage.resident ? 0 : *sc.io_workers;
        if (!stage.resident) {
            check(stage.prefix < sd.blocks.size() && stage.prefix >= sd.min_prefix,
                  sd.id + ": unsupported prefix");
            check(stage.slot_count >= sd.min_slots && stage.slot_count <= sd.max_slots,
                  sd.id + ": unsupported slot count");
            check(stage.group_size <= sd.max_group_size, sd.id + ": unsupported group size");
            check(stage.prefix == 0 || sd.blocks[stage.prefix - 1].safe_boundary_after,
                  "unsafe prefix boundary");
        }
        field(canonical, sd.id); field(canonical, sd.adapter_revision);
        canonical << stage.resident << ',' << stage.prefix << ',' << stage.group_size << ','
                  << stage.slot_count << ',' << stage.distance << ',' << stage.workers << ','
                  << stage.pass_count << '|';
        canonical << sd.passes.size() << '|';
        for (const auto &p : sd.passes) {
            identity(p.phase); field(canonical, p.phase); canonical << p.step << '|';
            shape(canonical, p.shape);
        }
        std::set<uint32_t> block_ids;
        std::map<std::string, std::vector<std::pair<std::string, uint64_t>>> classes;
        std::map<std::string, std::pair<uint64_t, uint64_t>> storages;
        std::map<std::string, std::optional<Materialization>> storage_materializations;
        std::map<std::string, std::string> class_materializations;
        std::set<std::string> resident_storages, streamed_storages;
        std::vector<std::optional<uint64_t>> source_bytes(sd.blocks.size(), uint64_t{0});
        bool prefix_sources_known = true;
        uint64_t prefix_source_bytes = 0;
        for (uint32_t bi = 0; bi < sd.blocks.size(); ++bi) {
            const auto &b = sd.blocks[bi];
            check(block_ids.insert(b.id).second, "duplicate block id");
            identity(b.layout_class);
            check(!b.fields.empty() && b.fields.size() <= max_fields_per_block, "field count limit");
            total_fields = add(total_fields, b.fields.size());
            check(total_fields <= max_descriptor_fields, "plan_too_large");
            std::vector<std::pair<std::string, uint64_t>> signature;
            std::set<std::string> names;
            std::map<std::string, const FieldSpec *> earlier_fields;
            std::ostringstream binding_signature;
            binding_signature.imbue(std::locale::classic());
            canonical << b.id << ',' << b.streamable << ',' << b.safe_boundary_after << '|';
            field(canonical, b.layout_class);
            for (const auto &f : b.fields) {
                identity(f.name); identity(f.storage_id);
                check(names.insert(f.name).second, "duplicate field name");
                check(f.bytes > 0, "empty field");
                const auto size = aligned(f.bytes, f.alignment);
                signature.emplace_back(f.name, f.alignment);
                const auto [si, fresh] = storages.emplace(f.storage_id, std::pair(f.bytes, f.alignment));
                check(fresh || si->second == std::pair(f.bytes, f.alignment), "alias layout mismatch");
                const auto [mi, new_materialization] = storage_materializations.emplace(f.storage_id, f.materialization);
                check(new_materialization || mi->second == f.materialization, "alias materialization mismatch");
                if (bi < stage.prefix) {
                    if (resident_storages.insert(f.storage_id).second)
                        stage.prefix_bytes = add(stage.prefix_bytes, size);
                } else {
                    // Refillable aliases need an explicit shared resident field protocol.
                    // Reject rather than silently duplicate or overwrite a shared tensor.
                    check(!resident_storages.contains(f.storage_id) &&
                          streamed_storages.insert(f.storage_id).second,
                          "streamed shared storage requires resident extraction");
                }
                field(canonical, f.name); field(canonical, f.storage_id);
                canonical << f.bytes << ',' << f.alignment << ',' << bool(f.materialization) << '|';
                binding_signature << bool(f.materialization) << '|';
                uint64_t read_bytes = 0;
                if (f.materialization) {
                    const auto &m = *f.materialization;
                    identity(m.format); identity(m.storage_mode); identity(m.conversion);
                    check(m.reads.size() <= 64, "field source count limit");
                    total_source_ranges = add(total_source_ranges, m.reads.size());
                    check(total_source_ranges <= max_descriptor_fields, "plan_too_large: source ranges");
                    check(m.derived_from.empty() != m.reads.empty(), "field requires reads OR derivation");
                    for (auto *stream : {&canonical, &binding_signature}) {
                        field(*stream, m.format); field(*stream, m.storage_mode); field(*stream, m.conversion);
                        shape(*stream, m.shape); field(*stream, m.derived_from);
                        *stream << m.derived_offset << '|';
                    }
                    if (!m.derived_from.empty()) {
                        identity(m.derived_from);
                        const auto dep = earlier_fields.find(m.derived_from);
                        check(dep != earlier_fields.end() && dep->second->materialization.has_value(),
                              "derived field must reference earlier materialized field");
                        check(m.derived_offset <= dep->second->bytes &&
                              f.bytes <= dep->second->bytes - m.derived_offset, "derived range overflow");
                    } else check(m.derived_offset == 0, "source field has derived offset");
                    canonical << m.reads.size() << '|';
                    for (const auto &r : m.reads) {
                        check(r.artifact < d.artifacts.size(), "unknown source artifact");
                        const auto &a = d.artifacts[r.artifact];
                        check(r.bytes && r.offset <= a.bytes && r.bytes <= a.bytes - r.offset,
                              "source range overflow");
                        identity(r.tensor); identity(r.dtype);
                        canonical << r.artifact << ',' << r.offset << ',' << r.bytes << '|';
                        field(canonical, r.tensor); field(canonical, r.dtype); shape(canonical, r.shape);
                        read_bytes = add(read_bytes, r.bytes);
                    }
                    if (source_bytes[bi]) *source_bytes[bi] = add(*source_bytes[bi], read_bytes);
                } else {
                    source_bytes[bi].reset(); out.materializations_complete = false;
                    if (bi < stage.prefix) prefix_sources_known = false;
                }
                if (bi < stage.prefix && fresh) prefix_source_bytes = add(prefix_source_bytes, read_bytes);
                earlier_fields.emplace(f.name, &f);
            }
            const auto [ci, fresh] = classes.emplace(b.layout_class, signature);
            check(fresh || ci->second == signature, "layout class field signature mismatch");
            const auto [bc, new_class] = class_materializations.emplace(b.layout_class, binding_signature.str());
            check(new_class || bc->second == binding_signature.str(), "layout class materialization mismatch");
        }
        if (prefix_sources_known) stage.prefix_source_read_bytes = prefix_source_bytes;
        stage.source_read_bytes_per_pass = uint64_t{0};
        for (uint32_t begin = stage.prefix; begin < sd.blocks.size();) {
            const auto &cls = sd.blocks[begin].layout_class;
            uint32_t end = begin + 1;
            while (end < sd.blocks.size() && sd.blocks[end].layout_class == cls) ++end;
            PoolLayout pool;
            pool.id = static_cast<uint32_t>(stage.pools.size()); pool.layout_class = cls;
            const uint32_t count = 1 + (end - begin - 1) / stage.group_size;
            check(stage.slot_count <= count, "slot_count_exceeds_groups: " + sd.id);
            const size_t field_count = sd.blocks[begin].fields.size();
            const size_t group_fields = size_t(std::min(stage.group_size, end - begin)) * field_count;
            check(group_fields <= max_descriptor_fields, "plan_too_large: group fields");
            pool.slots.resize(stage.slot_count);
            for (auto &slot : pool.slots) slot.field_capacity.resize(group_fields);
            uint32_t ordinal = 0;
            for (uint32_t at = begin; at < end;) {
                Group group;
                group.id = static_cast<uint32_t>(stage.groups.size());
                group.pool = pool.id; group.slot = ordinal++ % stage.slot_count;
                const uint32_t tail = at + std::min(stage.group_size, end - at);
                for (uint32_t bi = at; bi < tail; ++bi) {
                    const auto &b = sd.blocks[bi];
                    check(b.streamable, "block is not streamable");
                    group.blocks.push_back(b.id);
                    for (const auto &f : b.fields) {
                        const auto bytes = aligned(f.bytes, f.alignment);
                        group.field_bytes.push_back(bytes);
                        group.bytes = add(group.bytes, f.bytes);
                    }
                }
                check(sd.blocks[tail - 1].safe_boundary_after, "unsafe group boundary");
                auto &slot = pool.slots[group.slot];
                for (size_t fi = 0; fi < group.field_bytes.size(); ++fi)
                    slot.field_capacity[fi] = std::max(slot.field_capacity[fi], group.field_bytes[fi]);
                stage.suffix_content_bytes_per_pass = add(stage.suffix_content_bytes_per_pass, group.bytes);
                for (uint32_t bi = at; bi < tail; ++bi)
                    if (!source_bytes[bi]) stage.source_read_bytes_per_pass.reset();
                    else if (stage.source_read_bytes_per_pass)
                        *stage.source_read_bytes_per_pass = add(*stage.source_read_bytes_per_pass, *source_bytes[bi]);
                stage.groups.push_back(std::move(group));
                at = tail;
            }
            for (auto &slot : pool.slots) {
                for (auto bytes : slot.field_capacity) slot.capacity_bytes = add(slot.capacity_bytes, bytes);
                pool.capacity_bytes = add(pool.capacity_bytes, slot.capacity_bytes);
            }
            stage.peak_pool_bytes = std::max(stage.peak_pool_bytes, pool.capacity_bytes);
            stage.pools.push_back(std::move(pool));
            begin = end;
        }
        // Validate total request read counter before any execution.
        check(stage.suffix_content_bytes_per_pass <= std::numeric_limits<uint64_t>::max() / sd.pass_count,
              "pass byte count overflow");
        check(!stage.source_read_bytes_per_pass ||
              *stage.source_read_bytes_per_pass <= UINT64_MAX / sd.pass_count,
              "source pass byte count overflow");
        out.stages.push_back(std::move(stage));
    }
    for (const auto &[id, ignored] : c.stages) check(stages.contains(id), "unknown stage " + id);
    out.canonical = canonical.str();
    out.digest = memory_sha256_hex(out.canonical);
    return out;
}
} // namespace tc::streaming
