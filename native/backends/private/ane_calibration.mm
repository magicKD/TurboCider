#include "ane_calibration.hpp"

#include <algorithm>
#include <exception>
#include <set>
#include <cstring>
#include <unistd.h>

namespace tc::ane::private_api {
CalibrationBatch::CalibrationBatch(Device &device, Program &program,
                                   std::vector<CalibrationBindings> bindings, uint64_t &timeline)
    : device_(device) {
    if (bindings.empty() || bindings.size() > 4)
        throw std::invalid_argument("calibration batch requires one to four evaluations");
    // Reject accidental scratch reuse BEFORE issuing any driver work. The
    // returned ticket's surface owners, not these borrowed data addresses,
    // provide lifetime retention; addresses are only an alias check here.
    std::set<void *> inputs, outputs;
    for (const auto &row : bindings)
        for (const auto &input : row.inputs) inputs.insert(input.second.data());
    for (const auto &row : bindings) for (const auto &output : row.outputs)
        if (inputs.contains(output.second.data()) || !outputs.insert(output.second.data()).second)
            throw std::invalid_argument("calibration evaluations require independent nonaliased outputs");
    const auto previous = std::max(timeline, device.value());
    if (previous > UINT64_MAX - 2) throw std::overflow_error("calibration event timeline exhausted");
    ready_ = previous + 1;
    const auto done = previous + 2;
    prepared_.reserve(bindings.size());
    for (const auto &row : bindings) prepared_.push_back(program.prepare(row.inputs, row.outputs, ready_, done));
    timeline = done;
}

CalibrationBatchResult CalibrationBatch::measure(bool ane,
                                                 const std::function<void()> &submit_gpu,
                                                 const std::function<void()> &finish_gpu) {
    if (consumed_) throw std::invalid_argument("calibration batch already consumed");
    if (bool(submit_gpu) != bool(finish_gpu) || (!ane && !submit_gpu))
        throw std::invalid_argument("calibration requires ANE or complete GPU submission/drain callbacks");
    consumed_ = true;
    std::vector<Ticket> tickets;
    tickets.reserve(prepared_.size());
    // No driver evaluation has been submitted yet. Meet the dependency on
    // the CPU before timing, not via a signal CB hidden in an ANE-alone arm.
    if (ane) device_.release_prepared(ready_);
    const auto start = std::chrono::steady_clock::now();
    std::exception_ptr error;
    bool gpu_attempted = false;
    try {
        if (ane) for (auto &request : prepared_) tickets.push_back(request.submit());
        if (submit_gpu) { gpu_attempted = true; submit_gpu(); }
    } catch (...) { error = std::current_exception(); }
    // Never let a GPU error/cancellation free either engine's borrowed input
    // or scratch before all successfully submitted producers/consumers join.
    if (gpu_attempted) {
        try { finish_gpu(); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    CalibrationBatchResult result;
    result.ok = true;
    for (auto &ticket : tickets) {
        try {
            const auto completion = ticket.finish();
            ++result.ane_calls;
            if (!completion.ok) {
                result.ok = false;
                if (result.error.empty()) result.error = completion.error;
            }
        } catch (...) { if (!error) error = std::current_exception(); }
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // Unsubmitted PreparedRequests can now be discarded outside the clock.
    prepared_.clear();
    if (error) std::rethrow_exception(error);
    return result;
}

struct W8GpuCalibrationWork::Impl {
    struct Bank {
        Surface g, sg, u, su, d, sd;
        Bank(Device &device, const GraphShape &s)
            : g(device, s.width, s.hidden, Element::I8), sg(device, s.width, 1, Element::FP16),
              u(device, s.width, s.hidden, Element::I8), su(device, s.width, 1, Element::FP16),
              d(device, s.hidden, s.width, Element::I8), sd(device, s.hidden, 1, Element::FP16) {}
        uint64_t bytes() const { return g.bytes() + sg.bytes() + u.bytes() + su.bytes() + d.bytes() + sd.bytes(); }
    };
    Device device;
    GraphShape shape;
    std::vector<W8GpuCalibrationLayer> layers;
    std::array<std::unique_ptr<Bank>, 2> banks;
    std::array<std::optional<Surface>, 2> x, tx, dg, du;
    std::vector<QuantStage> stages;
    std::optional<Transfer> transfer;
    W8GpuCalibrationStats counters;
    uint64_t estimate = 0, allocated = 0;
    int count = 0, gpu_channels = 0;
    bool prepared = false, active = false, head_attempted = false, transfer_submitted = false;
    bool dynamic_corrections = false;
    std::vector<bool> correction_attempted;
    std::vector<std::pair<DeviceMatrixView,DeviceMatrixView>> correction_sources;
    std::exception_ptr submit_error;

    Impl(Device &d, GraphShape s, std::vector<W8GpuCalibrationLayer> source,
         MemoryLimits limits, uint64_t resident, uint64_t mlx_active)
        : device(d), shape(s), layers(std::move(source)) {
        if (shape.kind != Kind::SwiGLU || layers.size() != 5)
            throw std::invalid_argument("W8 GPU calibration requires SwiGLU and five depth/future sources");
        bool corrections = false;
        std::vector<CalibrationSurfaceShape> snapshots;
        auto describe = [&](const Surface &surface) {
            if (surface.element() != Element::FP16) throw std::invalid_argument("calibration restore snapshot must be FP16");
            snapshots.push_back({surface.rows(), surface.columns(), 2});
        };
        for (size_t index = 0; index < layers.size(); ++index) {
            const auto &layer = layers[index];
            const auto &g = layer.weights[0], &u = layer.weights[1], &w = layer.weights[2];
            const auto &a = layer.activation;
            const int first = g.selection.row_begin;
            if (first <= 0 || (gpu_channels && first != gpu_channels) ||
                g.source.rows != first + shape.width || u.source.rows != g.source.rows ||
                g.source.cols != shape.hidden || u.source.cols != shape.hidden ||
                w.source.rows != shape.hidden || w.source.cols != g.source.rows ||
                g.selection.rows != shape.width || g.selection.columns != shape.hidden || g.selection.column_begin != 0 ||
                u.selection.row_begin != first || u.selection.rows != shape.width || u.selection.column_begin != 0 ||
                u.selection.columns != shape.hidden || w.selection.row_begin != 0 || w.selection.rows != shape.hidden ||
                w.selection.column_begin != first || w.selection.columns != shape.width ||
                g.selection.rotation_block != 128 || u.selection.rotation_block != 128 || w.selection.rotation_block != 512 ||
                g.selection.transpose || u.selection.transpose || w.selection.transpose ||
                g.selection.rotation_seed != 20260930 || u.selection.rotation_seed != 20260930 || w.selection.rotation_seed != 20260930 ||
                (index < 4 && (a.rows != shape.rows || a.cols != shape.hidden || layer.restoration.empty())))
                throw std::invalid_argument("W8 calibration physical source/range/activation/restore geometry mismatch");
            gpu_channels = first;
            if(layer.correction_producer) {
                if(!shape.lora_inputs || !layer.correction_drain || layer.corrections || index>=4)
                    throw std::invalid_argument("dynamic calibration correction requires independent producer/drain and current LoRA layer");
                corrections=true;dynamic_corrections=true;
            }
            if (layer.corrections) {
                const auto &c = *layer.corrections;
                if (!shape.lora_inputs || c.first.rows != shape.rows || c.second.rows != shape.rows ||
                    c.first.cols != shape.width || c.second.cols != shape.width)
                    throw std::invalid_argument("W8 calibration correction geometry mismatch");
                corrections = true;
            }
            for (const auto &restore : layer.restoration) {
                if (restore.source.columns() != uint32_t(shape.rows))
                    throw std::invalid_argument("W8 calibration restore row count mismatch");
                describe(restore.source);
                if (restore.row_scales) describe(*restore.row_scales);
                if (restore.token_scales) describe(*restore.token_scales);
                if (restore.second_token_scales) describe(*restore.second_token_scales);
            }
        }
        const auto plan = plan_gpu_calibration_memory(shape.rows, shape.hidden, shape.width, corrections,
                                                       snapshots, uint64_t(getpagesize()));
        if (!plan) throw MemoryBudgetError("W8 GPU calibration surface plan invalid or overflowed");
        estimate = plan->estimated_bytes;
        const auto observed = observe_runtime_memory(mlx_active);
        const auto decision = admit_memory(observed, limits, resident, estimate);
        if (!decision.allowed()) throw MemoryBudgetError("W8 GPU calibration admission denied (" +
            memory_denial_reason(decision.denial, observed) + ")");
        for (auto &bank : banks) { bank = std::make_unique<Bank>(device, shape); allocated += bank->bytes(); }
        auto make = [&](std::optional<Surface> &target, int rows, int cols, Element element) {
            target.emplace(device, rows, cols, element); allocated += target->bytes();
        };
        for (int slot = 0; slot < 2; ++slot) {
            make(x[slot], shape.hidden, shape.rows, Element::I8);
            make(tx[slot], 1, shape.rows, Element::FP16);
            if (corrections) {
                make(dg[slot], shape.width, shape.rows, Element::FP16);
                make(du[slot], shape.width, shape.rows, Element::FP16);
            }
        }
        auto freeze = [&](const Surface &surface) {
            Surface copy(device, surface.rows(), surface.columns(), surface.element());
            if (copy.pitch() != surface.pitch()) throw std::invalid_argument("calibration snapshot pitch mismatch");
            std::memcpy(copy.data(), surface.data(), surface.rows() * surface.pitch());
            allocated += copy.bytes();
            return copy;
        };
        for (auto &layer : layers) for (auto &restore : layer.restoration) {
            restore.source = freeze(restore.source);
            if (restore.row_scales) restore.row_scales = freeze(*restore.row_scales);
            if (restore.token_scales) restore.token_scales = freeze(*restore.token_scales);
            if (restore.second_token_scales) restore.second_token_scales = freeze(*restore.second_token_scales);
        }
        if (allocated > plan->surface_bytes) throw MemoryBudgetError("W8 actual calibration surfaces exceed admission");
    }
    void stage_bank(int layer, int slot) {
        const auto &sources = layers.at(layer).weights;
        auto &b = *banks.at(slot);
        const std::array<std::pair<Surface, Surface>, 3> destinations{{{b.g, b.sg}, {b.u, b.su}, {b.d, b.sd}}};
        for (int i = 0; i < 3; ++i) {
            stages.push_back(device.stage_w8(sources[i].source, sources[i].selection,
                                             destinations[i].first, destinations[i].second));
            ++counters.weight_projections;
        }
    }
    void stage_activation(int layer, int slot) {
        const auto &a = layers.at(layer).activation;
        DeviceWeightView source{a.buffer, a.buffer_bytes, a.offset_bytes, a.row_stride_bytes,
            a.rows, a.cols, DeviceWeightEncoding::Dense, a.dtype, 32, {}, {}, a.owner};
        stages.push_back(device.stage_w8(std::move(source), {0, shape.rows, 0, shape.hidden, 128, 20260930, true},
                                         *x[slot], *tx[slot]));
        ++counters.activation_packs;
    }
    std::exception_ptr drain() noexcept {
        std::exception_ptr error;
        for(size_t layer=0;layer<correction_attempted.size();++layer)if(correction_attempted[layer]) {
            correction_attempted[layer]=false;
            try{layers[layer].correction_drain();}catch(...){if(!error)error=std::current_exception();}
        }
        for (auto &stage : stages) {
            try { const auto result = stage.finish(); if (!result.ok) throw CapabilityError(result.error); }
            catch (...) { if (!error) error = std::current_exception(); }
        }
        stages.clear();
        if (transfer_submitted && transfer) {
            try {
                const auto result = transfer->finish();
                if (!result.ok) throw CapabilityError(result.error);
                if (transfer->validation_flags()) throw CapabilityError("W8 calibration GPU restore/upload validation failed");
            } catch (...) { if (!error) error = std::current_exception(); }
        }
        return error;
    }
};

W8GpuCalibrationWork::W8GpuCalibrationWork(Device &device, GraphShape shape,
        std::vector<W8GpuCalibrationLayer> layers, MemoryLimits limits, uint64_t resident, uint64_t mlx_active)
    : impl_(std::make_unique<Impl>(device, shape, std::move(layers), limits, resident, mlx_active)) {}
W8GpuCalibrationWork::~W8GpuCalibrationWork() { impl_->drain(); }
void W8GpuCalibrationWork::prepare(int count, bool prefetch) {
    auto &p = *impl_;
    if (p.active || (count != 1 && count != 4)) throw std::invalid_argument("W8 calibration prepare requires idle one/four layers");
    if (const auto error = p.drain()) std::rethrow_exception(error);
    p.prepared = false; p.transfer.reset();
    p.count = count; p.counters = {}; p.counters.layers = count; p.counters.prefetch = prefetch;
    p.head_attempted = p.transfer_submitted = false; p.submit_error = {};
    p.correction_attempted.assign(size_t(count),false);p.correction_sources.clear();
    std::vector<Upload> uploads;
    std::vector<Download> downloads;
    for (int layer = 0; layer < count; ++layer) {
        const auto &source = p.layers.at(layer);
        if (source.corrections) {
            uploads.push_back({source.corrections->first, *p.dg[layer & 1], 0});
            uploads.push_back({source.corrections->second, *p.du[layer & 1], 0});
        }
        downloads.insert(downloads.end(), source.restoration.begin(), source.restoration.end());
    }
    p.counters.correction_uploads = uploads.size(); p.counters.restore_downloads = downloads.size();
    if(!p.dynamic_corrections) {
        p.transfer = p.device.prepare_gpu_transfer(std::move(uploads), std::move(downloads));
        p.counters.independent_gpu_transfer = p.transfer->independent_gpu();
    }
    p.stages.reserve(size_t(count + 1) * 3 + count);
    p.prepared = true;
}
void W8GpuCalibrationWork::submit(const Head &head) {
    auto &p = *impl_;
    if (!p.prepared || p.active || !head) throw std::invalid_argument("W8 calibration GPU work unprepared or missing head");
    p.prepared = false; p.active = true;
    try {
        if (p.counters.prefetch) p.stage_bank(0, 0);
        std::vector<Upload> uploads;
        std::vector<Download> downloads;
        for (int layer = 0; layer < p.count; ++layer) {
            if(p.dynamic_corrections) {
                const auto &source=p.layers.at(layer);
                if(source.correction_producer) {
                    p.correction_attempted[size_t(layer)]=true;
                    auto correction=source.correction_producer();
                    for(const auto *view:{&correction.first,&correction.second})
                        if(view->rows!=p.shape.rows || view->cols!=p.shape.width || !view->owner)
                            throw std::invalid_argument("dynamic calibration correction produced invalid owned geometry");
                    p.correction_sources.push_back(std::move(correction));
                    const auto &views=p.correction_sources.back();
                    uploads.push_back({views.first,*p.dg[layer&1],0});uploads.push_back({views.second,*p.du[layer&1],0});
                    ++p.counters.correction_computations;
                } else if(source.corrections) {
                    uploads.push_back({source.corrections->first,*p.dg[layer&1],0});
                    uploads.push_back({source.corrections->second,*p.du[layer&1],0});
                }
                downloads.insert(downloads.end(),source.restoration.begin(),source.restoration.end());
            }
            if (!p.counters.prefetch) p.stage_bank(layer, layer & 1);
            p.stage_activation(layer, layer & 1);
            p.head_attempted = true; head(layer, p.gpu_channels);
            if (p.counters.prefetch) p.stage_bank(layer + 1, (layer + 1) & 1);
        }
        if(p.dynamic_corrections) {
            p.counters.correction_uploads=uploads.size();
            p.transfer=p.device.prepare_gpu_transfer(std::move(uploads),std::move(downloads));
            p.counters.independent_gpu_transfer=p.transfer->independent_gpu();
        }
        p.transfer->submit(); p.transfer_submitted = true;
    } catch (...) { p.submit_error = std::current_exception(); throw; }
}
void W8GpuCalibrationWork::finish(const Fence &heads, const Join &join, const Fence &joins) {
    auto &p = *impl_;
    if (!p.active) throw std::invalid_argument("W8 calibration GPU work not active");
    std::exception_ptr error = p.submit_error;
    // Drain heads even if their submission callback threw partway through.
    if (p.head_attempted) {
        try { if (!heads) throw std::invalid_argument("calibration missing head fence"); heads(); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    if (const auto produced = p.drain(); produced && !error) error = produced;
    if (!error) {
        try {
            if (!join || !joins) throw std::invalid_argument("calibration missing join/fence");
            for (int layer = 0; layer < p.count; ++layer) { join(layer); ++p.counters.joins; }
        } catch (...) { error = std::current_exception(); }
        // A join callback may also throw after queuing earlier GPU work.
        if (joins) try { joins(); } catch (...) { if (!error) error = std::current_exception(); }
    }
    p.counters.completed = !error;
    p.active = false; p.submit_error = {}; p.transfer.reset();
    p.correction_sources.clear();
    if (error) std::rethrow_exception(error);
}
W8GpuCalibrationStats W8GpuCalibrationWork::stats() const { return impl_->counters; }
uint64_t W8GpuCalibrationWork::estimated_bytes() const { return impl_->estimate; }
uint64_t W8GpuCalibrationWork::allocated_surface_bytes() const { return impl_->allocated; }

ChannelCalibrationSamples measure_w8_channel_point(Device &device,Program &program,
    const std::array<std::vector<CalibrationBindings>,2> &bindings,W8GpuCalibrationWork &work,
    uint64_t &timeline,double share,bool prefetch,const std::function<void()> &reset,
    const W8GpuCalibrationWork::Head &head,const W8GpuCalibrationWork::Fence &heads,
    const W8GpuCalibrationWork::Join &join,const W8GpuCalibrationWork::Fence &joins,
    int warmups,int repeats) {
    if(!std::isfinite(share) || share<=0 || share>=1 || bindings[0].size()!=1 || bindings[1].size()!=4 ||
        !reset || !head || !heads || !join || !joins || !calibration_sampling_valid(warmups,repeats))
        throw std::invalid_argument("channel calibration requires complete one/four bindings, callbacks and bounded odd repeats");
    ChannelCalibrationSamples result;result.prefetch=prefetch;
    for(int sweep=0;sweep<warmups+repeats;++sweep) for(int position=0;position<6;++position) {
        const int cell=(sweep+position)%6,count_index=cell/3,part=cell%3,count=count_index?4:1;
        reset();
        if(part!=1)work.prepare(count,prefetch);
        CalibrationBatch batch(device,program,bindings[count_index],timeline);
        const auto measured=part==1?batch.measure(true):batch.measure(part==2,
            [&]{work.submit(head);},[&]{work.finish(heads,join,joins);});
        if(!measured.ok || !std::isfinite(measured.seconds) || measured.seconds<=0 ||
            measured.ane_calls!=uint64_t(part==0?0:count))
            throw CapabilityError("invalid channel calibration completion/count/timing");
        if(part!=1) {
            const auto traffic=work.stats();
            if(!traffic.completed || !traffic.independent_gpu_transfer || traffic.layers!=uint64_t(count) ||
                traffic.weight_projections!=uint64_t(3*(count+(prefetch?1:0))) || traffic.activation_packs!=uint64_t(count) ||
                traffic.restore_downloads<uint64_t(count) || traffic.joins!=uint64_t(count))
                throw CapabilityError("channel calibration lacks complete staging/pack/restore/join/future traffic");
            result.correction_computations+=traffic.correction_computations;
            result.correction_uploads+=traffic.correction_uploads;
        }
        result.ane_calls+=measured.ane_calls;
        if(sweep>=warmups)result.seconds[part][count_index].push_back(measured.seconds);
    }
    auto median=[](std::vector<double> values) {std::sort(values.begin(),values.end());return values[values.size()/2];};
    result.point={share,
        calibration_layer_seconds(median(result.seconds[0][0]),median(result.seconds[0][1])),
        calibration_layer_seconds(median(result.seconds[1][0]),median(result.seconds[1][1])),
        calibration_layer_seconds(median(result.seconds[2][0]),median(result.seconds[2][1]))};
    return result;
}
} // namespace tc::ane::private_api
