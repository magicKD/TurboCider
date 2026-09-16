#include "memory_manifest.hpp"

#include "../core/common.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>

namespace tc {
namespace {

bool valid_sha256(std::string_view value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

void append_field(std::ostringstream &out, std::string_view name,
                  std::string_view value) {
    out << '|' << name << '=' << value.size() << ':' << value;
}

template <typename T>
void append_number(std::ostringstream &out, std::string_view name, T value) {
    out << '|' << name << '=' << value;
}

uint32_t rotr(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
}

class Sha256 {
  public:
    void update(std::string_view input) {
        for (unsigned char byte : input) {
            buffer_[buffer_size_++] = byte;
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t original_bits = bit_count_ + buffer_size_ * 8ull;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) buffer_[buffer_size_++] = 0;
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) buffer_[buffer_size_++] = 0;
        for (int index = 7; index >= 0; --index)
            buffer_[buffer_size_++] =
                static_cast<uint8_t>(original_bits >> (index * 8));
        transform(buffer_.data());

        std::array<uint8_t, 32> result{};
        for (size_t word = 0; word < state_.size(); ++word) {
            result[word * 4 + 0] = static_cast<uint8_t>(state_[word] >> 24);
            result[word * 4 + 1] = static_cast<uint8_t>(state_[word] >> 16);
            result[word * 4 + 2] = static_cast<uint8_t>(state_[word] >> 8);
            result[word * 4 + 3] = static_cast<uint8_t>(state_[word]);
        }
        return result;
    }

  private:
    void transform(const uint8_t *block) {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        std::array<uint32_t, 64> words{};
        for (unsigned index = 0; index < 16; ++index) {
            words[index] = (uint32_t(block[index * 4]) << 24) |
                           (uint32_t(block[index * 4 + 1]) << 16) |
                           (uint32_t(block[index * 4 + 2]) << 8) |
                           uint32_t(block[index * 4 + 3]);
        }
        for (unsigned index = 16; index < 64; ++index) {
            const uint32_t s0 = rotr(words[index - 15], 7) ^
                rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotr(words[index - 2], 17) ^
                rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned index = 0; index < 64; ++index) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + k[index] + words[index];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> buffer_{};
    size_t buffer_size_ = 0;
    uint64_t bit_count_ = 0;
};

template <typename T>
void append_enum(std::ostringstream &out, std::string_view name, T value) {
    append_number(out, name, static_cast<unsigned>(value));
}

} // namespace

const char *allocation_lifetime_name(AllocationLifetime value) {
    switch (value) {
    case AllocationLifetime::Request: return "request";
    case AllocationLifetime::Stage: return "stage";
    case AllocationLifetime::Block: return "block";
    case AllocationLifetime::Tile: return "tile";
    case AllocationLifetime::CommandBuffer: return "command_buffer";
    case AllocationLifetime::Export: return "export";
    }
    return "request";
}

const char *upper_provenance_name(UpperProvenance value) {
    switch (value) {
    case UpperProvenance::ExactMetadata: return "exact_metadata";
    case UpperProvenance::ExactShapeFormula: return "exact_shape_formula";
    case UpperProvenance::ValidatedEnvelope: return "validated_envelope";
    case UpperProvenance::HeuristicNotExecutable:
        return "heuristic_not_executable";
    }
    return "heuristic_not_executable";
}

std::string memory_checkpoint_identity_digest(
        const std::vector<MemoryCheckpointFileIdentity> &files) {
    require(!files.empty(),
            "memory_checkpoint_invalid: checkpoint file list is empty");
    std::vector<const MemoryCheckpointFileIdentity *> sorted;
    sorted.reserve(files.size());
    for (const auto &file : files) sorted.push_back(&file);
    std::sort(sorted.begin(), sorted.end(), [](const auto *left,
                                               const auto *right) {
        return left->logical_name < right->logical_name;
    });
    std::ostringstream canonical;
    append_field(canonical, "schema", "turbocider.checkpoint_identity.v1");
    std::string previous;
    for (const auto *file : sorted) {
        require(!file->logical_name.empty(),
                "memory_checkpoint_invalid: logical file name is empty");
        require(previous.empty() || previous != file->logical_name,
                "memory_checkpoint_invalid: duplicate logical file " +
                    file->logical_name);
        require(file->size_bytes > 0,
                "memory_checkpoint_invalid: checkpoint file is empty " +
                    file->logical_name);
        require(valid_sha256(file->sha256),
                "memory_checkpoint_invalid: invalid file digest " +
                    file->logical_name);
        append_field(canonical, "logical_name", file->logical_name);
        append_number(canonical, "size_bytes", file->size_bytes);
        append_field(canonical, "sha256", file->sha256);
        previous = file->logical_name;
    }
    return memory_sha256_hex(canonical.str());
}

void MemoryManifest::validate() const {
    require(schema == "turbocider.memory_manifest.v1",
            "memory_manifest_invalid: unsupported schema");
    require(!candidate_id.empty(),
            "memory_manifest_invalid: candidate_id is empty");
    require(valid_sha256(checkpoint_digest),
            "memory_manifest_invalid: checkpoint digest is invalid");
    require(!backend_revision.empty(),
            "memory_manifest_invalid: backend revision is empty");
    require(!runtime_revision.empty(),
            "memory_manifest_invalid: runtime revision is empty");
    std::map<std::string, const AllocationSiteSpec *> sites_by_id;
    std::map<std::string, uint64_t> instance_count;
    std::map<std::pair<std::string, uint32_t>, bool> instance_ids;
    std::map<uint64_t, std::vector<const AllocationInstance *>> alias_members;
    for (const auto &site : sites) {
        require(!site.site_id.empty(),
                "memory_manifest_invalid: allocation site id is empty");
        require(sites_by_id.emplace(site.site_id, &site).second,
                "memory_manifest_invalid: duplicate allocation site " +
                    site.site_id);
        if (site.required)
            require(site.provenance != UpperProvenance::HeuristicNotExecutable,
                    "memory_estimate_unknown: required allocation site " +
                        site.site_id + " is heuristic");
    }
    for (const auto &instance : instances) {
        require(instance.upper_bytes > 0,
                "memory_manifest_invalid: allocation upper is zero");
        require(instance.live_begin < instance.live_end,
                "memory_manifest_invalid: allocation lifetime is empty");
        require(!instance.last_use_event.empty(),
                "memory_manifest_invalid: allocation last-use event is empty");
        require(sites_by_id.count(instance.site_id),
                "memory_manifest_invalid: orphan allocation instance " +
                    instance.site_id);
        const auto *site = sites_by_id.at(instance.site_id);
        require(instance_ids.emplace(
                    std::make_pair(instance.site_id, instance.instance_id),
                    true).second,
                "memory_manifest_invalid: duplicate allocation instance " +
                    instance.site_id);
        require(!instance.alias_group || site->aliasable,
                "memory_manifest_invalid: non-aliasable site declares alias " +
                    instance.site_id);
        if (instance.alias_group)
            alias_members[instance.alias_group].push_back(&instance);
        instance_count[instance.site_id]++;
    }
    for (auto &[group, members] : alias_members) {
        (void)group;
        std::sort(members.begin(), members.end(),
                  [](const auto *left, const auto *right) {
                      return std::tie(left->live_begin, left->live_end,
                                      left->site_id, left->instance_id) <
                             std::tie(right->live_begin, right->live_end,
                                      right->site_id, right->instance_id);
                  });
        for (size_t index = 1; index < members.size(); ++index)
            require(members[index - 1]->live_end <=
                        members[index]->live_begin,
                    "memory_manifest_invalid: alias lifetimes overlap for " +
                        members[index]->site_id);
    }
    for (const auto &[site_id, site] : sites_by_id)
        if (site->required)
            require(instance_count[site_id] > 0,
                    "memory_estimate_unknown: missing required allocation "
                    "instance " + site_id);
}

const AllocationSiteSpec *MemoryManifest::find_site(
        std::string_view site_id) const {
    for (const auto &site : sites)
        if (site.site_id == site_id) return &site;
    return nullptr;
}

MemoryManifestBuilder::MemoryManifestBuilder(
        std::string candidate_id, std::string checkpoint_digest,
        std::string backend_revision, std::string runtime_revision) {
    manifest_.candidate_id = std::move(candidate_id);
    manifest_.checkpoint_digest = std::move(checkpoint_digest);
    manifest_.backend_revision = std::move(backend_revision);
    manifest_.runtime_revision = std::move(runtime_revision);
}

void MemoryManifestBuilder::add_site(AllocationSiteSpec site) {
    require(!built_, "memory_manifest_invalid: builder already finalized");
    require(!site.site_id.empty(),
            "memory_manifest_invalid: allocation site id is empty");
    require(std::none_of(
                manifest_.sites.begin(), manifest_.sites.end(),
                [&](const auto &existing) {
                    return existing.site_id == site.site_id;
                }),
            "memory_manifest_invalid: duplicate allocation site " +
                site.site_id);
    manifest_.sites.push_back(std::move(site));
}

void MemoryManifestBuilder::add_instance(AllocationInstance instance) {
    require(!built_, "memory_manifest_invalid: builder already finalized");
    require(!instance.site_id.empty() && instance.upper_bytes > 0,
            "memory_manifest_invalid: invalid allocation instance");
    manifest_.instances.push_back(std::move(instance));
}

MemoryManifest MemoryManifestBuilder::build() {
    require(!built_, "memory_manifest_invalid: builder already finalized");
    manifest_.validate();
    built_ = true;
    return std::move(manifest_);
}

std::string MemoryManifest::digest() const {
    validate();
    std::vector<const AllocationSiteSpec *> sorted_sites;
    for (const auto &site : sites) sorted_sites.push_back(&site);
    std::sort(sorted_sites.begin(), sorted_sites.end(),
              [](const auto *a, const auto *b) { return a->site_id < b->site_id; });
    std::vector<const AllocationInstance *> sorted_instances;
    for (const auto &instance : instances) sorted_instances.push_back(&instance);
    std::sort(sorted_instances.begin(), sorted_instances.end(),
              [](const auto *a, const auto *b) {
                  return std::tie(a->site_id, a->instance_id) <
                         std::tie(b->site_id, b->instance_id);
              });
    std::ostringstream canonical;
    append_field(canonical, "schema", schema);
    append_field(canonical, "candidate", candidate_id);
    append_field(canonical, "checkpoint", checkpoint_digest);
    append_field(canonical, "backend_revision", backend_revision);
    append_field(canonical, "runtime_revision", runtime_revision);
    for (const auto *site : sorted_sites) {
        append_field(canonical, "site_id", site->site_id);
        append_field(canonical, "component", site->component);
        append_field(canonical, "stage", site->stage);
        append_enum(canonical, "memory_class", site->memory_class);
        append_enum(canonical, "lifetime", site->lifetime);
        append_enum(canonical, "provenance", site->provenance);
        append_number(canonical, "required", site->required ? 1 : 0);
        append_number(canonical, "asynchronous", site->asynchronous ? 1 : 0);
        append_number(canonical, "aliasable", site->aliasable ? 1 : 0);
        append_number(canonical, "guard_threshold", site->guard_threshold_bytes);
    }
    for (const auto *instance : sorted_instances) {
        append_field(canonical, "instance_site", instance->site_id);
        append_number(canonical, "instance_id", instance->instance_id);
        append_number(canonical, "upper", instance->upper_bytes);
        append_number(canonical, "live_begin", instance->live_begin);
        append_number(canonical, "live_end", instance->live_end);
        append_number(canonical, "alias_group", instance->alias_group);
        append_field(canonical, "last_use", instance->last_use_event);
    }
    return memory_sha256_hex(canonical.str());
}

bool MemoryCandidateKey::operator==(const MemoryCandidateKey &other) const {
    return std::tie(adapter, model_id, checkpoint_digest, backend, dtype,
                    model_variant, operation, shape_bucket, sampler_mode,
                    refill_slots, tiling_mode, runtime_revision,
                    device_family) ==
           std::tie(other.adapter, other.model_id, other.checkpoint_digest,
                    other.backend, other.dtype, other.model_variant,
                    other.operation, other.shape_bucket, other.sampler_mode,
                    other.refill_slots, other.tiling_mode,
                    other.runtime_revision, other.device_family);
}

bool MemoryCandidateKey::operator<(const MemoryCandidateKey &other) const {
    return std::tie(adapter, model_id, checkpoint_digest, backend, dtype,
                    model_variant, operation, shape_bucket, sampler_mode,
                    refill_slots, tiling_mode, runtime_revision,
                    device_family) <
           std::tie(other.adapter, other.model_id, other.checkpoint_digest,
                    other.backend, other.dtype, other.model_variant,
                    other.operation, other.shape_bucket, other.sampler_mode,
                    other.refill_slots, other.tiling_mode,
                    other.runtime_revision, other.device_family);
}

std::string MemoryCandidateKey::canonical() const {
    std::ostringstream out;
    append_field(out, "adapter", adapter);
    append_field(out, "model", model_id);
    append_field(out, "checkpoint", checkpoint_digest);
    append_field(out, "backend", backend);
    append_field(out, "dtype", dtype);
    append_field(out, "variant", model_variant);
    append_field(out, "operation", operation);
    append_field(out, "shape", shape_bucket);
    append_field(out, "sampler", sampler_mode);
    append_number(out, "refill_slots", refill_slots);
    append_field(out, "tiling", tiling_mode);
    append_field(out, "runtime", runtime_revision);
    append_field(out, "device", device_family);
    return out.str();
}

std::string MemoryCandidateKey::digest() const {
    return memory_sha256_hex(canonical());
}

void MemoryCapabilityRegistry::add(MemoryCapabilityRecord record) {
    require(!record.key.adapter.empty(),
            "memory_capability_invalid: adapter is empty");
    require(!record.manifest_digest.empty(),
            "memory_capability_invalid: manifest digest is empty");
    require(valid_sha256(record.manifest_digest),
            "memory_capability_invalid: manifest digest is invalid");
    require(valid_sha256(record.key.checkpoint_digest),
            "memory_capability_invalid: checkpoint digest is invalid");
    require(record.level >= MemoryCapabilityLevel::EnvelopeValidated,
            "memory_capability_invalid: record is not executable");
    require(record.maximum_validated_upper_bytes > 0,
            "memory_capability_invalid: validated upper is zero");
    require(record.framework_upper_bytes > 0,
            "memory_capability_invalid: framework upper is zero");
    require(!record.framework_provenance.empty(),
            "memory_capability_invalid: framework provenance is empty");
    require(record.framework_upper_bytes <=
                record.maximum_validated_upper_bytes,
            "memory_capability_invalid: framework upper exceeds validated upper");
    require(record.require_explicit_epoch,
            "memory_capability_invalid: executable record must require explicit epochs");
    require(!record.require_explicit_schedule ||
                record.require_explicit_epoch,
            "memory_capability_invalid: explicit schedule requires explicit epochs");
    require(!record.require_explicit_schedule || !record.schedule.empty(),
            "memory_capability_invalid: explicit schedule is absent");
    require(record.schedule.empty() || record.require_explicit_schedule,
            "memory_capability_invalid: schedule bindings are not required");
    if (record.state == MemoryCertificationState::Certified) {
        require(record.level == MemoryCapabilityLevel::L3Certified &&
                    record.release_enabled &&
                    valid_sha256(record.evidence_digest),
                "memory_capability_invalid: certified record is incomplete");
    } else {
        require(record.state ==
                    MemoryCertificationState::ExperimentalGuarded &&
                    !record.release_enabled,
                "memory_capability_invalid: non-certified record is invalid");
    }
    auto duplicate = std::find_if(records_.begin(), records_.end(),
                                  [&](const auto &item) {
                                      return item.key == record.key &&
                                             item.manifest_digest ==
                                                 record.manifest_digest;
                                  });
    require(duplicate == records_.end(),
            "memory_capability_invalid: duplicate capability record");
    records_.push_back(std::move(record));
}

MemoryCapabilityMatch MemoryCapabilityRegistry::lookup(
        const MemoryCandidateKey &key, std::string_view manifest_digest,
        bool allow_experimental) const {
    MemoryCapabilityMatch result;
    bool key_seen = false;
    for (const auto &record : records_) {
        if (!(record.key == key)) continue;
        key_seen = true;
        if (record.manifest_digest != manifest_digest) continue;
        if (!allow_experimental &&
            (record.state != MemoryCertificationState::Certified ||
             !record.release_enabled))
            continue;
        if (allow_experimental &&
            record.state != MemoryCertificationState::Certified &&
            record.state != MemoryCertificationState::ExperimentalGuarded)
            continue;
        if (record.level < MemoryCapabilityLevel::EnvelopeValidated) continue;
        result.matched = true;
        result.reason = "capability manifest matched";
        result.record = record;
        return result;
    }
    result.reason = key_seen ?
        "capability manifest or release identity mismatch" :
        "no capability record matches candidate key";
    return result;
}

std::string memory_sha256_hex(std::string_view value) {
    Sha256 hash;
    hash.update(value);
    const auto bytes = hash.finish();
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) output << std::setw(2) << unsigned(byte);
    return output.str();
}

} // namespace tc
