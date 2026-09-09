#include "../platform/apple/bridge.hpp"
#include "../backends/mlx.hpp"
#include "../backends/coreml.hpp"
#include "turbocider/turbocider.h"
#include <mutex>
#include <cstring>
#include "../runtime/execution.hpp"
#include "../models/ltx_runtime/ltx_gemma_tokenizer.h"
#include "../models/ltx_runtime/ltx_weights.h"
#import <Metal/Metal.h>
#include <cmath>
struct tc_engine {
    std::unique_ptr<tc::ModelSession> session;
    std::mutex mutex;
    std::atomic<bool> cancelled{false};
};
struct tc_coreml_ffn {
    std::unique_ptr<tc::HybridSession> session;
    std::mutex mutex;
};
namespace {
using tc::configure_streams;
using tc::DeviceLease;
char *copy(const std::string &s) {
    auto p = strdup(s.c_str());
    if (!p)
        throw std::bad_alloc();
    return p;
}
int fail(char **error, const std::exception &e) {
    if (error)
        *error = strdup(e.what());
    return dynamic_cast<const tc::Cancelled *>(&e) ? 2 : 1;
}
} // namespace

uint32_t tc_abi_version(void) {
    return 1;
}
void tc_string_free(char *s) {
    free(s);
}
int tc_coreml_ffn_create(const char *manifest, const char *checkpoint,
                         int minimum_rows, int warmups,
                         tc_coreml_ffn **bridge, char **error) {
    if (bridge)
        *bridge = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(manifest && checkpoint && bridge,
                        "missing Core ML FFN manifest/checkpoint or output handle");
            tc::require(minimum_rows > 0 && minimum_rows <= 8192,
                        "invalid Core ML FFN minimum row count");
            tc::require(warmups >= 0 && warmups <= 8,
                        "invalid Core ML FFN warmup count");
            configure_streams();
            auto value = std::make_unique<tc_coreml_ffn>();
            std::atomic<bool> cancelled{false};
            tc::Event event = [](const std::string &, int, int) {};
            value->session = std::make_unique<tc::HybridSession>(
                std::filesystem::path(manifest), std::filesystem::path{}, minimum_rows,
                event, cancelled, warmups, std::filesystem::path(checkpoint));
            *bridge = value.release();
            return 0;
        } catch (const std::exception &exception) {
            return fail(error, exception);
        } catch (...) {
            if (error)
                *error = strdup("unknown Core ML FFN creation error");
            return 1;
        }
    }
}
int tc_coreml_ffn_predict(tc_coreml_ffn *bridge, int block,
                          const uint16_t *input, int rows,
                          uint16_t *output, char **error) {
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(bridge && bridge->session && input && output,
                        "missing Core ML FFN bridge or buffer");
            std::lock_guard<std::mutex> lock(bridge->mutex);
            auto metrics = bridge->session->metrics();
            tc::require(rows > 0 && rows <= metrics.bucket,
                        "Core ML FFN request exceeds the manifest row bucket");
            tc::require(block >= 0 && block < metrics.block_count,
                        "Core ML FFN block index is out of range");
            // `input` contains IEEE FP16 bit patterns.  The iterator
            // constructor would numerically convert each uint16_t value to
            // FP16 (for example 0x3400 -> 13312) instead of reinterpreting the
            // bits.  Wrap the caller-owned storage explicitly for the
            // duration of this synchronous prediction.
            auto value = tc::Tensor(
                const_cast<uint16_t *>(input), {1, rows, metrics.hidden},
                tc::mx::float16, [](void *) {});
            if (rows < metrics.bucket)
                value = tc::mx::concatenate(
                    {value, tc::mx::zeros({1, metrics.bucket - rows, metrics.hidden},
                                         tc::mx::float16)},
                    1);
            tc::mx::eval(value);
            auto prediction = bridge->session->predict(block, value);
            prediction = tc::slice_axis(prediction, 1, 0, rows);
            tc::mx::eval(prediction);
            std::memcpy(output, prediction.data<tc::mx::float16_t>(),
                        size_t(rows) * size_t(metrics.hidden) * sizeof(uint16_t));
            return 0;
        } catch (const std::exception &exception) {
            return fail(error, exception);
        } catch (...) {
            if (error)
                *error = strdup("unknown Core ML FFN prediction error");
            return 1;
        }
    }
}
int tc_coreml_ffn_metrics_json(tc_coreml_ffn *bridge, char **result, char **error) {
    if (result)
        *result = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(bridge && bridge->session && result,
                        "missing Core ML FFN bridge or result pointer");
            std::lock_guard<std::mutex> lock(bridge->mutex);
            *result = copy(tc::json(tc::to_dictionary(bridge->session->metrics())));
            return 0;
        } catch (const std::exception &exception) {
            return fail(error, exception);
        } catch (...) {
            if (error)
                *error = strdup("unknown Core ML FFN metrics error");
            return 1;
        }
    }
}
void tc_coreml_ffn_free(tc_coreml_ffn *bridge) {
    delete bridge;
}
char *tc_system_json(void) {
    @autoreleasepool {
        try {
            return copy(tc::json(tc::system_info()));
        } catch (...) {
            return strdup("{\"error\":\"system probe failed\"}");
        }
    }
}
char *tc_models_json(void) {
    @autoreleasepool {
        try {
            return copy(tc::json(tc::to_dictionary(tc::describe_modules())));
        } catch (...) {
            return strdup("{}");
        }
    }
}
int tc_plan_json(const char *r, char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(out, "missing output pointer");
            *out = copy(tc::json(
                tc::to_dictionary(tc::make_plan(tc::request_from_json(tc::parse_json(r))))));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error)
                *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_engine_create_model(const char *id, const char *path, tc_engine **engine, char **error) {
    if (engine)
        *engine = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(id && path && engine, "missing model id/path or output handle");
            auto &module = tc::module_for(id);
            tc::require(bool(module.create), "model executor unavailable");
            auto e = std::make_unique<tc_engine>();
            e->session = module.create(path);
            *engine = e.release();
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error)
                *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_engine_create(const char *path, tc_engine **engine, char **error) {
    return tc_engine_create_model("flux2-klein-4b", path, engine, error);
}
extern "C" int tc_engine_create_model_candidate(const char *id, const char *path,
                                                tc_engine **engine, char **error) {
    if (!id || std::strcmp(id, "ltx-2.5-distilled") != 0) {
        if (engine) *engine = nullptr;
        if (error) *error = strdup("candidate executor is restricted to LTX");
        return 1;
    }
    return tc_engine_create_model(id, path, engine, error);
}
void tc_engine_cancel(tc_engine *e) {
    if (e)
        e->cancelled.store(true);
}
void tc_engine_free(tc_engine *e) {
    delete e;
}
int tc_engine_generate(tc_engine *e, const char *r, tc_event_callback cb, void *ctx, char **out,
                       char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(e && out, "missing engine or output");
            std::unique_lock<std::mutex> local(e->mutex, std::try_to_lock);
            tc::require(local.owns_lock(), "engine busy");
            // MLX uses process-global device/allocation policy. Serialize all embeddings.
            std::unique_lock<std::mutex> global(tc::execution_mutex(), std::try_to_lock);
            tc::require(global.owns_lock(), "native GPU runtime busy");
            DeviceLease device_lease;
            e->cancelled.store(false);
            auto request = tc::request_from_json(tc::parse_json(r));
            const bool parent_mlx = e->session->uses_parent_mlx(request);
            if (parent_mlx) {
                tc::require(tc::mx::is_available(tc::mx::Device(tc::mx::Device::gpu)),
                            "Metal GPU unavailable");
                configure_streams();
            }
            struct Drain {
                bool enabled;
                ~Drain() {
                    if (enabled) {
                        try {
                            tc::mx::synchronize();
                        } catch (...) {
                        }
                    }
                }
            } drain{parent_mlx};
            uint64_t seq = 0;
            auto begin = tc::Clock::now();
            tc::Event event = [&](const std::string &phase, int current, int total) {
                if (!(phase == "export" && current == total))
                    tc::checkpoint(e->cancelled);
                if (cb) {
                    @autoreleasepool {
                        auto s = tc::json(@{
                            @"schema_version" : @1,
                            @"sequence" : @(++seq),
                            @"phase" : @(phase.c_str()),
                            @"completed" : @(current),
                            @"total" : @(total),
                            @"elapsed_seconds" :
                                @(std::chrono::duration<double>(tc::Clock::now() - begin).count())
                        });
                        cb(s.c_str(), ctx);
                    }
                }
            };
            auto result = e->session->generate(request, event, e->cancelled);
            *out = copy(tc::json(tc::to_dictionary(result)));
            return 0;
        } catch (const std::exception &ex) {
            return fail(error, ex);
        } catch (...) {
            if (error)
                *error = strdup("unknown native error");
            return 1;
        }
    }
}
int tc_tokenize_json(const char *path, const char *prompt, char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(path && prompt && out, "missing tokenizer input");
            tc::Tokenizer t(std::filesystem::path(path) / "tokenizer");
            auto ids = t.prompt(prompt);
            NSMutableArray *a = [NSMutableArray array];
            for (int i : ids.ids)
                [a addObject:@(i)];
            *out = copy(tc::json(@{@"ids" : a, @"valid" : @(ids.valid)}));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            return 1;
        }
    }
}
int tc_ltx_gemma_tokenize_json(const char *tokenizer_json, const char *prompt,
                               uint32_t max_length, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(tokenizer_json && prompt && out, "missing Gemma tokenizer input");
            char detail[1024] = {};
            ltx_gemma_tokenizer *tokenizer =
                ltx_gemma_tokenizer_load(tokenizer_json, detail, sizeof(detail));
            tc::require(tokenizer != nullptr, detail);
            uint32_t *ids = nullptr;
            uint8_t *mask = nullptr;
            size_t count = 0;
            int ok = ltx_gemma_tokenizer_encode(tokenizer, prompt, max_length, &ids, &mask,
                                                &count, detail, sizeof(detail));
            ltx_gemma_tokenizer_free(tokenizer);
            tc::require(ok, detail);
            NSMutableArray *id_array = [NSMutableArray arrayWithCapacity:count];
            NSMutableArray *mask_array = [NSMutableArray arrayWithCapacity:count];
            for (size_t index = 0; index < count; ++index) {
                [id_array addObject:@(ids[index])];
                [mask_array addObject:@(mask[index])];
            }
            ltx_gemma_tokenizer_ids_free(ids, mask);
            *out = copy(tc::json(@{@"ids" : id_array, @"mask" : mask_array,
                                   @"count" : @(count)}));
            return 0;
        } catch (const std::exception &e) { return fail(error, e); }
        catch (...) { if (error) *error = strdup("unknown native error"); return 1; }
    }
}
int tc_ltx_gemma_inspect_json(const char *checkpoint, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(checkpoint && out, "missing Gemma checkpoint input");
            char detail[1024] = {};
            ltx_gemma_checkpoint_info info{};
            tc::require(ltx_gemma_checkpoint_inspect(checkpoint, &info, detail, sizeof(detail)), detail);
            tc::require(ltx_gemma_checkpoint_validate(&info, detail, sizeof(detail)), detail);
            *out = copy(tc::json(@{
                @"tensor_count" : @(info.tensor_count), @"vocab_size" : @(info.vocab_size),
                @"hidden_size" : @(info.hidden_size), @"intermediate_size" : @(info.intermediate_size),
                @"layers" : @(info.num_layers), @"attention_heads" : @(info.attention_heads),
                @"sliding_kv_heads" : @(info.sliding_kv_heads), @"full_kv_heads" : @(info.full_kv_heads),
                @"sliding_head_dim" : @(info.sliding_head_dim), @"full_head_dim" : @(info.full_head_dim),
                @"projection_input_dim" : @(info.projection_input_dim),
                @"projection_video_dim" : @(info.projection_video_dim),
                @"projection_audio_dim" : @(info.projection_audio_dim),
                @"attention_k_eq_v" : @(info.attention_k_eq_v != 0),
                @"quantization" : @"int8_convrot_256", @"validated" : @YES
            }));
            return 0;
        } catch (const std::exception &e) { return fail(error, e); }
        catch (...) { if (error) *error = strdup("unknown native error"); return 1; }
    }
}
static int lora_preflight(const char *model_path, const char *lora_path, float strength,
                          bool fastmetal, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(model_path && lora_path && out, "missing LoRA preflight input");
            tc::require(std::isfinite(strength) && strength > 0.0f && strength <= 4.0f,
                        "LoRA strength must be in (0, 4]");
            tc::LoRAAsset lora{lora_path, strength, "transformer"};
            auto result = fastmetal ? tc::preflight_wan_lora(model_path, lora)
                                    : tc::preflight_ltx_lora(model_path, lora);
            *out = copy(tc::json(result));
            return 0;
        } catch (const std::exception &e) { return fail(error, e); }
        catch (...) { if (error) *error = strdup("unknown native error"); return 1; }
    }
}
int tc_ltx_lora_preflight_json(const char *model_path, const char *lora_path,
                               float strength, char **out, char **error) {
    return lora_preflight(model_path, lora_path, strength, false, out, error);
}
int tc_fastmetal_lora_preflight_json(const char *model_path, const char *lora_path,
                                     float strength, char **out, char **error) {
    return lora_preflight(model_path, lora_path, strength, true, out, error);
}
int tc_wan_lora_preflight_json(const char *model_path, const char *lora_path,
                              float strength, char **out, char **error) {
    return lora_preflight(model_path, lora_path, strength, true, out, error);
}
int tc_ltx_audio_preflight_json(const char *model_path, char **out, char **error) {
    if (out) *out = nullptr;
    if (error) *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(model_path && out, "missing LTX audio preflight input");
            *out = copy(tc::json(tc::preflight_ltx_audio(model_path)));
            return 0;
        } catch (const std::exception &e) { return fail(error, e); }
        catch (...) { if (error) *error = strdup("unknown native error"); return 1; }
    }
}
int tc_native_self_test(char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(out, "missing output");
            std::lock_guard<std::mutex> lock(tc::execution_mutex());
            using namespace tc;
            for (auto &s : {"flux2-klein-4b", "ltx-2.5-distilled", "minimax-h3-turbo"})
                validate_recipe(model_recipe(s));
            bool rejected = false;
            try {
                validate_recipe({"bad", {{"a", {"b"}}}, false});
            } catch (...) {
                rejected = true;
            }
            require(rejected, "graph cycle/missing dependency accepted");
            auto sigmas = flux_sigmas(1024, 4);
            require(sigmas.front() == 1 && sigmas.back() == 0 && sigmas[1] > sigmas[2],
                    "bad schedule");
            configure_streams();
            auto x =
                mx::astype(mx::random::normal({4097}, mx::float32, 0.f, 1.f, mx::random::key(17)),
                           mx::bfloat16);
            auto noise = mx::astype(mx::sin(mx::astype(x, mx::float32)), mx::bfloat16);
            auto compiled_step = mx::compile(
                [](const std::vector<Tensor> &a) {
                    return std::vector<Tensor>{a[0] + a[2] * a[1]};
                },
                true);
            for (float dt : {-.25f, -.04191464f, -.71749657f, .0317f}) {
                auto step = euler_step(x, noise, dt);
                auto ref = compiled_step({x, noise, mx::astype(Tensor(dt), x.dtype())})[0];
                require(mx::all(step == ref).item<bool>(),
                        "custom Metal Euler mismatch for dt=" + std::to_string(dt));
            }
            auto y = norm(mx::reshape(mx::arange(64, mx::float32), {4, 16}));
            require(mx::max(mx::abs(mx::mean(y, -1))).item<float>() < 1e-5,
                    "normalization mismatch");
            *out = copy(json(@{
                @"passed" : @YES,
                @"checks" : @[
                    @"three_model_recipes", @"dependency_rejection", @"flux_schedule",
                    @"custom_metal_euler_bf16", @"layer_norm"
                ],
                @"system" : system_info()
            }));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            return 1;
        }
    }
}

int tc_compile_coreml_json(const char *source, const char *cache, char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(source && cache && out, "missing compile input/output");
            *out = copy(tc::json(tc::compile_artifact(source, cache)));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error)
                *error = strdup("unknown compile error");
            return 1;
        }
    }
}

int tc_engine_load(tc_engine *e, tc_event_callback cb, void *ctx, char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(e && out, "missing engine or output");
            std::unique_lock<std::mutex> local(e->mutex, std::try_to_lock);
            tc::require(local.owns_lock(), "engine busy");
            std::unique_lock<std::mutex> global(tc::execution_mutex(), std::try_to_lock);
            tc::require(global.owns_lock(), "native GPU runtime busy");
            DeviceLease lease;
            const bool parent_mlx = e->session->uses_parent_mlx();
            if (parent_mlx)
                configure_streams();
            e->cancelled.store(false);
            auto begin = tc::Clock::now();
            uint64_t sequence = 0;
            tc::Event event = [&](const std::string &phase, int current, int total) {
                tc::checkpoint(e->cancelled);
                if (cb) {
                    @autoreleasepool {
                        auto text = tc::json(@{
                            @"sequence" : @(++sequence),
                            @"phase" : @(phase.c_str()),
                            @"completed" : @(current),
                            @"total" : @(total),
                            @"elapsed_seconds" :
                                @(std::chrono::duration<double>(tc::Clock::now() - begin).count())
                        });
                        cb(text.c_str(), ctx);
                    }
                }
            };
            try {
                auto result = e->session->load(event, e->cancelled);
                if (parent_mlx)
                    tc::mx::synchronize();
                *out = copy(tc::json(tc::to_dictionary(result)));
            } catch (...) {
                if (parent_mlx)
                    tc::mx::synchronize();
                e->session->unload();
                if (parent_mlx)
                    tc::mx::clear_cache();
                throw;
            }
            return 0;
        } catch (const std::exception &ex) {
            return fail(error, ex);
        } catch (...) {
            if (error)
                *error = strdup("unknown load error");
            return 1;
        }
    }
}
int tc_engine_unload(tc_engine *e, char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(e && out, "missing engine or output");
            std::unique_lock<std::mutex> local(e->mutex, std::try_to_lock);
            tc::require(local.owns_lock(), "engine busy");
            std::unique_lock<std::mutex> global(tc::execution_mutex(), std::try_to_lock);
            tc::require(global.owns_lock(), "native GPU runtime busy");
            const bool parent_mlx = e->session->uses_parent_mlx();
            if (parent_mlx) {
                configure_streams();
                tc::mx::synchronize();
            }
            e->session->unload();
            if (parent_mlx)
                tc::mx::clear_cache();
            *out = copy(tc::json(@{
                @"released" : @YES,
                @"mlx_active_bytes" : @(parent_mlx ? tc::mx::get_active_memory() : 0),
                @"scope" : parent_mlx ? @"MLX allocator; excludes OS/file cache"
                                        : @"external native backend released"
            }));
            return 0;
        } catch (const std::exception &ex) {
            return fail(error, ex);
        } catch (...) {
            if (error)
                *error = strdup("unknown unload error");
            return 1;
        }
    }
}

static int preparation_call(tc_engine *e, const char *request, int warmup, bool cache,
                            tc_event_callback cb, void *ctx, char **out, char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(e && request && out, "missing preparation input");
            std::unique_lock<std::mutex> local(e->mutex, std::try_to_lock);
            tc::require(local.owns_lock(), "engine busy");
            std::unique_lock<std::mutex> global(tc::execution_mutex(), std::try_to_lock);
            tc::require(global.owns_lock(), "native GPU runtime busy");
            DeviceLease lease;
            std::optional<tc::Request> parsed_request;
            if (!cache)
                parsed_request = tc::request_from_json(tc::parse_json(request));
            const bool parent_mlx = cache || e->session->uses_parent_mlx(*parsed_request);
            if (parent_mlx)
                configure_streams();
            e->cancelled.store(false);
            struct Drain {
                bool enabled;
                ~Drain() {
                    if (enabled) {
                        try {
                            tc::mx::synchronize();
                        } catch (...) {
                        }
                    }
                }
            } drain{parent_mlx};
            auto begin = tc::Clock::now();
            uint64_t sequence = 0;
            tc::Event event = [&](const std::string &phase, int current, int total) {
                tc::checkpoint(e->cancelled);
                if (cb) {
                    auto text = tc::json(@{
                        @"sequence" : @(++sequence),
                        @"phase" : @(phase.c_str()),
                        @"completed" : @(current),
                        @"total" : @(total),
                        @"elapsed_seconds" :
                            @(std::chrono::duration<double>(tc::Clock::now() - begin).count())
                    });
                    cb(text.c_str(), ctx);
                }
            };
            NSDictionary *result;
            if (cache) {
                auto value = tc::parse_json(request);
                if ([value[@"action"] isEqual:@"clear"]) {
                    tc::mx::synchronize();
                    e->session->unload();
                    tc::mx::clear_cache();
                }
                result = tc::manage_coreml_cache(value, event, e->cancelled);
            } else {
                tc::require(warmup == 0 || warmup == 1, "warmup must be 0 or 1");
                auto r = std::move(*parsed_request);
                r.dump.clear();
                result =
                    tc::to_dictionary(e->session->prepare(r, warmup != 0, event, e->cancelled));
            }
            *out = copy(tc::json(result));
            return 0;
        } catch (const std::exception &ex) {
            return fail(error, ex);
        } catch (...) {
            if (error)
                *error = strdup("unknown preparation error");
            return 1;
        }
    }
}
int tc_engine_prepare(tc_engine *e, const char *r, int warmup, tc_event_callback cb, void *ctx,
                      char **out, char **error) {
    return preparation_call(e, r, warmup, false, cb, ctx, out, error);
}
int tc_engine_cache(tc_engine *e, const char *r, tc_event_callback cb, void *ctx, char **out,
                    char **error) {
    return preparation_call(e, r, 0, true, cb, ctx, out, error);
}

static std::atomic<bool> resource_cancelled{false};
void tc_coreml_resources_cancel() {
    resource_cancelled.store(true);
}
int tc_coreml_resources_json(const char *request, tc_event_callback cb, void *ctx, char **out,
                             char **error) {
    if (out)
        *out = nullptr;
    if (error)
        *error = nullptr;
    @autoreleasepool {
        try {
            tc::require(request && out, "missing resource request");
            auto parsed = tc::parse_json(request);
            const bool inventory = tc::string_value(parsed, @"action") == "inventory";
            std::unique_lock<std::mutex> global(tc::execution_mutex(), std::defer_lock);
            std::unique_ptr<DeviceLease> lease;
            if (!inventory) {
                tc::require(global.try_lock(), "runtime busy");
                lease = std::make_unique<DeviceLease>();
            }
            std::atomic<bool> inventory_cancelled{false};
            auto &cancelled = inventory ? inventory_cancelled : resource_cancelled;
            cancelled.store(false);
            auto begin = tc::Clock::now();
            uint64_t sequence = 0;
            tc::Event event = [&](const std::string &phase, int current, int total) {
                tc::checkpoint(cancelled);
                if (cb) {
                    auto text = tc::json(@{
                        @"sequence" : @(++sequence),
                        @"phase" : @(phase.c_str()),
                        @"completed" : @(current),
                        @"total" : @(total),
                        @"elapsed_seconds" :
                            @(std::chrono::duration<double>(tc::Clock::now() - begin).count())
                    });
                    cb(text.c_str(), ctx);
                }
            };
            *out = copy(
                tc::json(tc::coreml_resources(parsed, event, cancelled)));
            return 0;
        } catch (const std::exception &e) {
            return fail(error, e);
        } catch (...) {
            if (error)
                *error = strdup("unknown resource error");
            return 1;
        }
    }
}
