#pragma once

// Diagnostic request-local capture only. No MLX, Metal, model or runtime S1
// dependency: the caller supplies bounded, already-produced FP16 row batches.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace tc::ane::calibration {
inline constexpr size_t input_limit_bytes = 8 * 1024 * 1024;
inline constexpr size_t statistics_limit_bytes = 2 * 1024 * 1024;
inline constexpr size_t total_limit_bytes = 24 * 1024 * 1024;
inline constexpr size_t serialization_reserve_bytes = 8 * 1024 * 1024;
inline constexpr const char *capture_recipe = "ffn-input-fp16-regions-outliers-v1";

struct LoRAIdentity { std::string id, fingerprint; double strength = 0; };
struct Metadata {
    std::string request_id, model_id, model_fingerprint, recipe, identity_kind;
    std::vector<LoRAIdentity> loras;
    uint64_t seed = 0;
    int width = 0, height = 0, total_steps = 0, reference_size = 0, reference_count = 0;
    std::string execution_route;
    // Configuration and upstream execution provenance are not placement or
    // quantization qualification. Unknown is explicit and remains acceptable.
    std::string runtime_recipe = "unknown", configured_backend = "unknown", actual_execution = "unknown";
};
struct Config {
    bool enabled = false;
    std::vector<int> layers{0, 15, 31}, steps{0, 2, 5};
    size_t rows_per_point = 64;
    size_t input_budget = input_limit_bytes, statistics_budget = statistics_limit_bytes,
           total_budget = total_limit_bytes;
};
struct Region { std::string name; size_t begin = 0, end = 0; };
struct Point { int layer = 0, step = 0; std::string phase; std::vector<Region> regions; };
struct Record {
    Point point;
    size_t rows = 0, hidden = 0, rows_observed = 0;
    std::vector<float> channel_max;
    std::vector<size_t> sample_rows;
    std::vector<uint16_t> samples;
};
struct WeightStatistics { int layer; std::string source_fingerprint; std::vector<float> channel_max; };

inline void require(bool value, const char *message) {
    if (!value) throw std::invalid_argument(message);
}
inline bool valid_text(const std::string &value) {
    return !value.empty() && value.size() <= 256 &&
        std::none_of(value.begin(), value.end(), [](unsigned char c) { return c < 32; });
}
inline std::string quote(const std::string &value) {
    std::string result = "\"";
    for (const auto c : value) {
        if (c == '\\' || c == '"') result += '\\';
        result += c;
    }
    return result + '"';
}
inline float decode_fp16(uint16_t bits) {
    const int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    require(exponent != 31, "calibration input contains nonfinite FP16");
    const float magnitude = exponent ? std::ldexp(float(1024 + mantissa), exponent - 25) :
                                      std::ldexp(float(mantissa), -24);
    return (bits & 0x8000) ? -magnitude : magnitude;
}

class Sampler {
public:
    Sampler() = default;
    Sampler(Config config, Metadata metadata) : config_(std::move(config)), metadata_(std::move(metadata)) {
        if (!config_.enabled) return;
        require(valid_text(metadata_.request_id) && valid_text(metadata_.model_id) &&
                valid_text(metadata_.model_fingerprint) && valid_text(metadata_.recipe),
                "calibration requires complete request/model/recipe identity");
        require(metadata_.identity_kind == "validated-loader-identity" ||
                metadata_.identity_kind == "canonical-stat-identity" || metadata_.identity_kind == "checkpoint-sha256",
                "calibration requires explicit model identity kind");
        require(metadata_.width > 0 && metadata_.width <= 4096 && metadata_.height > 0 && metadata_.height <= 4096 &&
                metadata_.total_steps > 0 && metadata_.total_steps <= 4096 && metadata_.reference_count >= 0 &&
                metadata_.reference_count <= 8 && metadata_.reference_size >= 0 && metadata_.reference_size <= 4096 &&
                (!metadata_.reference_count || metadata_.reference_size > 0) && valid_text(metadata_.execution_route),
                "calibration requires complete generation provenance");
        require(valid_text(metadata_.runtime_recipe) && valid_text(metadata_.configured_backend) &&
                valid_text(metadata_.actual_execution), "invalid calibration runtime provenance");
        require(metadata_.loras.size() <= 16, "too many calibration LoRA identities");
        for (const auto &lora : metadata_.loras)
            require(valid_text(lora.id) && valid_text(lora.fingerprint) && std::isfinite(lora.strength),
                    "calibration requires complete finite LoRA identity");
        const auto selection = [](const auto &values, size_t count, int maximum) {
            return !values.empty() && values.size() <= count &&
                std::all_of(values.begin(), values.end(), [&](int v) { return v >= 0 && v <= maximum; });
        };
        require(selection(config_.layers,32,31) && selection(config_.steps,3,4096),
                "calibration selection exceeds 32 layers/3 steps");
        require(std::all_of(config_.steps.begin(), config_.steps.end(), [&](int step) { return step < metadata_.total_steps; }),
                "calibration selected step exceeds request steps");
        for (const auto *values : {&config_.layers, &config_.steps})
            for (size_t i = 0; i < values->size(); ++i)
                require(std::find(values->begin(), values->begin() + i, (*values)[i]) == values->begin() + i,
                        "duplicate calibration selection");
        require(config_.rows_per_point > 0 && config_.rows_per_point <= 64 &&
                config_.input_budget > 0 && config_.input_budget <= input_limit_bytes &&
                config_.statistics_budget > 0 && config_.statistics_budget <= statistics_limit_bytes &&
                config_.total_budget > serialization_reserve_bytes && config_.total_budget <= total_limit_bytes,
                "calibration budgets exceed hard limits");
        records_.reserve(config_.layers.size() * config_.steps.size());
    }
    bool enabled() const { return config_.enabled; }
    bool wants(int layer, int step) const {
        return enabled() && std::find(config_.layers.begin(), config_.layers.end(), layer) != config_.layers.end() &&
            std::find(config_.steps.begin(), config_.steps.end(), step) != config_.steps.end() &&
            std::none_of(records_.begin(), records_.end(), [&](const Record &r) {
                return r.point.layer == layer && r.point.step == step;
            });
    }
    bool complete() const { return enabled() && records_.size() == config_.layers.size() * config_.steps.size() &&
                                  weights_.size() == config_.layers.size(); }
    size_t input_bytes() const { return input_bytes_; }
    size_t statistics_bytes() const { return statistics_bytes_; }
    const std::vector<Record> &records() const { return records_; }
    bool needs_weight_channel_max(int layer) const {
        return enabled() && std::find(config_.layers.begin(), config_.layers.end(), layer) != config_.layers.end() &&
            std::none_of(weights_.begin(), weights_.end(), [&](const WeightStatistics &w) { return w.layer == layer; });
    }

    // Caller reduces all output rows of the selected layer's loaded base Wg/Wu
    // and transfers only their combined H-channel abs max, never model weights.
    // It may use existing validated loader/stat identity instead of hashing a
    // checkpoint. LoRA remains a separate request binding, not merged here.
    bool set_weight_channel_max(int layer, std::span<const float> maximum, std::string source_fingerprint) {
        if (!enabled() || std::find(config_.layers.begin(), config_.layers.end(), layer) == config_.layers.end()) return false;
        require(valid_text(source_fingerprint) && !maximum.empty() && maximum.size() <= 8192 &&
                std::all_of(maximum.begin(), maximum.end(), [](float v) { return std::isfinite(v) && v >= 0; }),
                "invalid selected weight channel statistics");
        for (const auto &weight : weights_) if (weight.layer == layer) {
            require(weight.source_fingerprint == source_fingerprint && weight.channel_max.size() == maximum.size() &&
                    std::equal(maximum.begin(), maximum.end(), weight.channel_max.begin()), "weight statistics identity changed");
            return false;
        }
        for (const auto &record : records_) if (record.point.layer == layer)
            require(record.hidden == maximum.size(), "weight/input hidden geometry mismatch");
        const size_t bytes = maximum.size() * sizeof(float);
        require(bytes <= config_.statistics_budget - statistics_bytes_ &&
                input_bytes_ + statistics_bytes_ + bytes + serialization_reserve_bytes + 64 * 1024 <= config_.total_budget,
                "selected weight statistics exceed calibration budget");
        weights_.push_back({layer, std::move(source_fingerprint), {maximum.begin(), maximum.end()}});
        statistics_bytes_ += bytes;
        return true;
    }

    // Reader(begin,count,dst) must write exactly count*hidden FP16 bits. Its
    // temporary storage must be limited to this batch; never pull all reference
    // rows into a new host matrix. The sampler reads each source row once.
    template<class Reader>
    bool observe_rows(Point point, size_t rows, size_t hidden, Reader &&reader) {
        if (!wants(point.layer, point.step)) return false;
        require(point.phase == "denoise" || point.phase == "prefill", "invalid calibration phase");
        require(rows > 0 && rows <= 32768 && hidden > 0 && hidden <= 8192,
                "invalid bounded calibration geometry");
        for (const auto &weight : weights_) if (weight.layer == point.layer)
            require(weight.channel_max.size() == hidden, "weight/input hidden geometry mismatch");
        require(!point.regions.empty() && point.regions.size() <= 8, "calibration requires row regions");
        size_t end = 0;
        for (size_t i = 0; i < point.regions.size(); ++i) {
            const auto &region = point.regions[i];
            require(valid_text(region.name) && region.begin == end && region.end > region.begin && region.end <= rows,
                    "calibration regions must exactly partition source rows");
            for (size_t j = 0; j < i; ++j) require(point.regions[j].name != region.name, "duplicate calibration region name");
            end = region.end;
        }
        require(end == rows, "calibration regions omit source rows");
        const size_t count = std::min(rows, config_.rows_per_point), bytes = count * hidden * 2;
        const size_t stats = hidden * sizeof(float), batch_rows = std::min<size_t>(128, rows);
        // Count both sampler scratch and the caller's one-batch transfer.
        const size_t workspace = (2 * batch_rows + count) * hidden * 2 + count * sizeof(double) +
                                 count * sizeof(size_t) + 64 * 1024;
        require(bytes <= config_.input_budget - input_bytes_, "calibration input budget exceeded");
        require(stats <= config_.statistics_budget - statistics_bytes_, "calibration statistics budget exceeded");
        require(input_bytes_ + bytes + statistics_bytes_ + stats + workspace + serialization_reserve_bytes <=
                config_.total_budget, "calibration total workspace budget exceeded");
        Record record{std::move(point), rows, hidden, 0, std::vector<float>(hidden), {}, {}};
        require(count >= record.point.regions.size(), "too few sample rows to cover regions");
        const size_t outliers = rows > count ? std::min({size_t(16), count / 4, count - record.point.regions.size()}) : 0;
        const size_t regular_count = count - outliers;
        require(regular_count >= record.point.regions.size(), "too few sample rows to cover regions");
        std::vector<size_t> regular;
        regular.reserve(regular_count);
        // Equal region representation avoids dropping short text segments next
        // to many reference rows. Fill any duplicate/short-region gaps globally.
        for (size_t region_index = 0; region_index < record.point.regions.size(); ++region_index) {
            const auto &region = record.point.regions[region_index];
            const size_t n = std::min(region.end - region.begin,
                regular_count / record.point.regions.size() + (region_index < regular_count % record.point.regions.size()));
            for (size_t i = 0; i < n; ++i)
                regular.push_back(region.begin + (n == 1 ? 0 : i * (region.end - region.begin - 1) / (n - 1)));
        }
        for (size_t i = 0; regular.size() < regular_count && i < rows; ++i)
            if (std::find(regular.begin(), regular.end(), i) == regular.end()) regular.push_back(i);
        std::sort(regular.begin(), regular.end());
        record.sample_rows = regular;
        record.sample_rows.reserve(count);
        record.samples.reserve(count * hidden);
        record.samples.resize(regular_count * hidden);
        struct Extreme { double peak; size_t row; std::vector<uint16_t> values; };
        std::vector<Extreme> extremes;
        extremes.reserve(outliers);
        std::vector<uint16_t> batch(batch_rows * hidden);
        for (size_t first = 0; first < rows; first += batch_rows) {
            const size_t n = std::min(batch_rows, rows - first);
            reader(first, n, std::span<uint16_t>(batch.data(), n * hidden));
            for (size_t local = 0; local < n; ++local) {
                const size_t row = first + local;
                const auto source = std::span<const uint16_t>(batch.data() + local * hidden, hidden);
                double peak = 0;
                for (size_t col = 0; col < hidden; ++col) {
                    const float absolute = std::abs(decode_fp16(source[col]));
                    record.channel_max[col] = std::max(record.channel_max[col], absolute);
                    peak = std::max(peak, double(absolute));
                }
                const auto found = std::lower_bound(regular.begin(), regular.end(), row);
                if (found != regular.end() && *found == row)
                    std::copy(source.begin(), source.end(), record.samples.begin() + (found - regular.begin()) * hidden);
                else if (outliers) {
                    const auto smallest = std::min_element(extremes.begin(), extremes.end(), [](const Extreme &a, const Extreme &b) {
                        return a.peak < b.peak;
                    });
                    if (extremes.size() < outliers) extremes.push_back({peak, row, {source.begin(), source.end()}});
                    else if (peak > smallest->peak) *smallest = {peak, row, {source.begin(), source.end()}};
                }
            }
            record.rows_observed += n;
        }
        std::sort(extremes.begin(), extremes.end(), [](const Extreme &a, const Extreme &b) { return a.row < b.row; });
        for (const auto &extreme : extremes) {
            record.sample_rows.push_back(extreme.row);
            record.samples.insert(record.samples.end(), extreme.values.begin(), extreme.values.end());
        }
        require(record.samples.size() == count * hidden && record.rows_observed == rows,
                "incomplete calibration source scan");
        input_bytes_ += bytes; statistics_bytes_ += stats;
        records_.push_back(std::move(record));
        return true;
    }
    bool observe_fp16(Point point, size_t rows, size_t hidden, std::span<const uint16_t> input) {
        if (!wants(point.layer, point.step)) return false;
        require(hidden && rows <= std::numeric_limits<size_t>::max() / hidden && input.size() == rows * hidden,
                "calibration input span geometry mismatch");
        return observe_rows(std::move(point), rows, hidden, [&](size_t first, size_t count, std::span<uint16_t> out) {
            std::copy_n(input.begin() + first * hidden, count * hidden, out.begin());
        });
    }

    // An exclusive new directory preserves existing results and symlinks. The
    // JSON has no runtime loading ABI; this is explicitly an unqualified capture.
    void write(const std::filesystem::path &directory) const {
        require(enabled() && !records_.empty() && directory.is_absolute(), "invalid calibration output request");
        const auto manifest = json(), weight_stats = weight_json();
        require(manifest.size() + weight_stats.size() <= serialization_reserve_bytes &&
                manifest.size() + weight_stats.size() + input_bytes_ <= config_.total_budget,
                "calibration output budget exceeded");
        require(std::filesystem::create_directory(directory), "calibration output directory already exists");
        std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
        for (size_t i = 0; i < records_.size(); ++i) {
            std::vector<uint8_t> bytes(records_[i].samples.size() * 2);
            for (size_t j = 0; j < records_[i].samples.size(); ++j) {
                bytes[j * 2] = uint8_t(records_[i].samples[j]); bytes[j * 2 + 1] = uint8_t(records_[i].samples[j] >> 8);
            }
            write_new(directory / ("input-" + std::to_string(i) + ".f16"), bytes.data(), bytes.size());
        }
        write_new(directory / "weight_stats.json", weight_stats.data(), weight_stats.size());
        write_new(directory / "capture.json", manifest.data(), manifest.size());
    }
    std::string json() const {
        std::ostringstream out; out << std::setprecision(9);
        out << "{\"schema_version\":1,\"artifact_type\":\"bounded_ffn_input_calibration\","
            << "\"capture_recipe\":" << quote(capture_recipe)
            << ",\"performance_sample\":false,\"quantization_qualified\":false,\"complete\":" << (complete() ? "true" : "false")
            << ",\"binding\":" << binding_json() << ",\"selection\":{\"layers\":[";
        for (size_t i = 0; i < config_.layers.size(); ++i) { if (i) out << ','; out << config_.layers[i]; }
        out << "],\"steps\":[";
        for (size_t i = 0; i < config_.steps.size(); ++i) { if (i) out << ','; out << config_.steps[i]; }
        out << "],\"rows_per_point\":" << config_.rows_per_point << "},\"budgets\":{\"input_bytes\":" << config_.input_budget
            << ",\"statistics_bytes\":" << config_.statistics_budget << ",\"total_bytes\":" << config_.total_budget
            << "},\"used\":{\"input_bytes\":" << input_bytes_ << ",\"statistics_bytes\":" << statistics_bytes_ << "},\"points\":[";
        for (size_t i = 0; i < records_.size(); ++i) {
            const auto &r = records_[i]; if (i) out << ',';
            out << "{\"layer\":" << r.point.layer << ",\"step\":" << r.point.step << ",\"phase\":" << quote(r.point.phase)
                << ",\"rows\":" << r.rows << ",\"hidden\":" << r.hidden << ",\"rows_observed\":" << r.rows_observed
                << ",\"dtype\":\"float16_le\",\"sample_file\":\"input-" << i << ".f16\",\"regions\":[";
            for (size_t j = 0; j < r.point.regions.size(); ++j) {
                const auto &region = r.point.regions[j]; if (j) out << ',';
                out << "{\"name\":" << quote(region.name) << ",\"begin\":" << region.begin << ",\"end\":" << region.end << '}';
            }
            out << "],\"sample_rows\":[";
            for (size_t j = 0; j < r.sample_rows.size(); ++j) { if (j) out << ','; out << r.sample_rows[j]; }
            out << "],\"channel_max\":[";
            for (size_t j = 0; j < r.channel_max.size(); ++j) { if (j) out << ','; out << r.channel_max[j]; }
            out << "]}";
        }
        out << "]}\n"; return out.str();
    }
    std::string weight_json() const {
        std::ostringstream out; out << std::setprecision(9);
        out << "{\"schema_version\":1,\"artifact_type\":\"ffn_weight_channel_max\","
            << "\"statistics_scope\":\"base_gate_up_all_rows\",\"binding\":" << binding_json() << ",\"layers\":[";
        for (size_t i = 0; i < weights_.size(); ++i) {
            const auto &w = weights_[i]; if (i) out << ',';
            out << "{\"layer\":" << w.layer << ",\"hidden\":" << w.channel_max.size()
                << ",\"source_fingerprint\":" << quote(w.source_fingerprint) << ",\"gate_up_channel_max\":[";
            for (size_t j = 0; j < w.channel_max.size(); ++j) { if (j) out << ','; out << w.channel_max[j]; }
            out << "]}";
        }
        out << "]}\n"; return out.str();
    }
private:
    std::string binding_json() const {
        std::ostringstream out; out << std::setprecision(17);
        out << "{\"request_id\":" << quote(metadata_.request_id) << ",\"model_id\":" << quote(metadata_.model_id)
            << ",\"model_fingerprint\":" << quote(metadata_.model_fingerprint) << ",\"identity_kind\":" << quote(metadata_.identity_kind)
            << ",\"recipe\":" << quote(metadata_.recipe) << ",\"seed\":" << metadata_.seed
            << ",\"width\":" << metadata_.width << ",\"height\":" << metadata_.height << ",\"total_steps\":" << metadata_.total_steps
            << ",\"reference_size\":" << metadata_.reference_size << ",\"reference_count\":" << metadata_.reference_count
            << ",\"execution_route\":" << quote(metadata_.execution_route) << ",\"runtime_recipe\":" << quote(metadata_.runtime_recipe)
            << ",\"configured_backend\":" << quote(metadata_.configured_backend) << ",\"actual_execution\":" << quote(metadata_.actual_execution)
            << ",\"loras\":[";
        for (size_t i = 0; i < metadata_.loras.size(); ++i) {
            const auto &l = metadata_.loras[i]; if (i) out << ',';
            out << "{\"id\":" << quote(l.id) << ",\"fingerprint\":" << quote(l.fingerprint) << ",\"strength\":" << l.strength << '}';
        }
        out << "]}"; return out.str();
    }
    static void write_new(const std::filesystem::path &path, const void *data, size_t bytes) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
        if (fd < 0) throw std::runtime_error("cannot exclusively create calibration file");
        size_t done = 0;
        while (done < bytes) {
            const auto wrote = ::write(fd, static_cast<const char *>(data) + done, bytes - done);
            if (wrote <= 0) { ::close(fd); throw std::runtime_error("calibration file write failed"); }
            done += size_t(wrote);
        }
        if (::close(fd)) throw std::runtime_error("calibration file close failed");
    }
    Config config_;
    Metadata metadata_;
    std::vector<Record> records_;
    std::vector<WeightStatistics> weights_;
    size_t input_bytes_ = 0, statistics_bytes_ = 0;
};
} // namespace tc::ane::calibration
