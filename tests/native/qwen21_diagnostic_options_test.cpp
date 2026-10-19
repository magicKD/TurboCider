#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <array>
#include <cassert>
#include <optional>

using tc::qwen21::W8A8CallBudget;
using tc::qwen21::expected_w8a8_calls;

namespace {
constexpr std::array cache_environment{
    "TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC",
    "TURBOCIDER_QWEN21_DBCACHE_THRESHOLD",
    "TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE",
    "TURBOCIDER_QWEN21_DBCACHE_FRONT_BLOCKS",
    "TURBOCIDER_QWEN21_DBCACHE_WARMUP_STEPS",
};
struct CacheEnvironment {
    std::array<std::optional<std::string>, cache_environment.size()> saved;
    CacheEnvironment() {
        for (size_t i = 0; i < saved.size(); ++i) {
            if (const auto *value = std::getenv(cache_environment[i])) saved[i] = value;
            assert(unsetenv(cache_environment[i]) == 0);
        }
    }
    ~CacheEnvironment() {
        for (size_t i = 0; i < saved.size(); ++i) {
            if (saved[i]) setenv(cache_environment[i], saved[i]->c_str(), 1);
            else unsetenv(cache_environment[i]);
        }
    }
};

void test_cache_options() {
    using tc::qwen21::db_cache_options;
    CacheEnvironment environment;
    tc::Request request;
    const auto disabled = db_cache_options(request);
    assert(!disabled.enabled && !disabled.diagnostic && disabled.mode == "off");
    assert(disabled.threshold == .08f && disabled.max_consecutive == 2);
    assert(disabled.front_blocks == 8 && disabled.warmup_steps == 8);
    for (const auto &mode : {"conservative", "balanced", "fast"}) {
        request.qwen21_dit_cache = mode;
        const auto options = db_cache_options(request);
        assert(options.enabled && !options.diagnostic && options.mode == mode);
        assert(options.threshold == (options.mode == "conservative" ? .15f : .25f));
        assert(options.max_consecutive == (options.mode == "conservative" ? 1 :
                                          options.mode == "balanced" ? 2 : 4));
        assert(options.front_blocks == 8 && options.warmup_steps == 8);
    }
    auto rejected = [&] {
        try { (void)db_cache_options(request); }
        catch (const std::invalid_argument &) { return true; }
        return false;
    };
    request.qwen21_dit_cache = "balanced";
    assert(setenv(cache_environment[0], "0", 1) == 0);
    assert(db_cache_options(request).enabled);
    assert(setenv(cache_environment[0], "1", 1) == 0);
    assert(rejected());
    assert(unsetenv(cache_environment[0]) == 0);
    for (const auto *name : {cache_environment[1], cache_environment[2],
                             cache_environment[3], cache_environment[4]}) {
        assert(setenv(name, "0", 1) == 0);
        assert(rejected()); // Even a zero numeric override is not a preset input.
        assert(setenv(name, "8", 1) == 0);
        assert(rejected()); // Matching preset geometry is still an override.
        assert(unsetenv(name) == 0);
    }
    request.qwen21_dit_cache = "unknown";
    assert(rejected());
    request.qwen21_dit_cache = "off";
    assert(setenv(cache_environment[0], "1", 1) == 0);
    assert(setenv(cache_environment[1], "0.25", 1) == 0);
    assert(setenv(cache_environment[2], "4", 1) == 0);
    const auto diagnostic = db_cache_options(request);
    assert(diagnostic.enabled && diagnostic.diagnostic && diagnostic.mode == "off");
    assert(diagnostic.threshold == .25f && diagnostic.max_consecutive == 4);
    assert(diagnostic.front_blocks == 8 && diagnostic.warmup_steps == 8);
    assert(setenv(cache_environment[1], "0.24", 1) == 0);
    assert(setenv(cache_environment[2], "3", 1) == 0);
    assert(setenv(cache_environment[3], "1", 1) == 0);
    assert(setenv(cache_environment[4], "4", 1) == 0);
    const auto sglang_candidate = db_cache_options(request);
    assert(sglang_candidate.enabled && sglang_candidate.diagnostic);
    assert(sglang_candidate.threshold == .24f && sglang_candidate.max_consecutive == 3);
    assert(sglang_candidate.front_blocks == 1 && sglang_candidate.warmup_steps == 4);
    request.qwen21_dit_cache_explicit = true;
    assert(!db_cache_options(request).enabled); // Explicit Off overrides every diagnostic knob.
    assert(setenv(cache_environment[3], "invalid", 1) == 0);
    assert(!db_cache_options(request).enabled); // A stale malformed env cannot defeat Off.
    request.qwen21_dit_cache_explicit = false;
    assert(rejected());
    assert(setenv(cache_environment[3], "1", 1) == 0);
    assert(setenv(cache_environment[4], "5", 1) == 0);
    assert(rejected());
    assert(setenv(cache_environment[4], "4", 1) == 0);
    assert(setenv(cache_environment[0], "0", 1) == 0);
    assert(rejected()); // Retain the established max-requires-opt-in contract.
    assert(unsetenv(cache_environment[2]) == 0);
    assert(rejected()); // Front/warmup also require the diagnostic flag.
    assert(unsetenv(cache_environment[3]) == 0);
    assert(unsetenv(cache_environment[4]) == 0);
    assert(!db_cache_options(request).enabled);
    assert(setenv(cache_environment[1], "nan", 1) == 0);
    assert(rejected());
    assert(unsetenv(cache_environment[1]) == 0);
    assert(setenv(cache_environment[0], "true", 1) == 0);
    assert(rejected());
}
void test_frozen_conservative_cache_request() {
    using tc::qwen21::frozen_conservative_cache_request;
    tc::Request r;
    r.model = "qwen-image-2.1";
    r.qwen21_dit_cache = "conservative";
    r.execution = "gpu_ane";
    r.qwen21_w8a8 = true;
    r.allow_approximation = true;
    r.ane_manifest = "verified-at-load.json";
    r.steps = 40;
    assert(frozen_conservative_cache_request(r));
    const auto valid = r;
    auto reject = [&](auto change) {
        auto other = valid;
        change(other);
        assert(!frozen_conservative_cache_request(other));
    };
    reject([](auto &x) { x.model = "other"; });
    reject([](auto &x) { x.qwen21_dit_cache = "off"; });
    reject([](auto &x) { x.qwen21_dit_cache = "balanced"; });
    reject([](auto &x) { x.qwen21_dit_cache = "fast"; });
    reject([](auto &x) { x.execution = "gpu"; });
    reject([](auto &x) { x.qwen21_w8a8 = false; });
    reject([](auto &x) { x.qwen21_gpu_w8a16 = true; });
    reject([](auto &x) { x.qwen21_gpu_full_ffn_blocks = {3, 5, 7}; });
    reject([](auto &x) { x.hybrid_mlp_mode = "runtime"; });
    reject([](auto &x) { x.qwen21_denoiser_backend = "native_metal"; });
    reject([](auto &x) { x.allow_approximation = false; });
    reject([](auto &x) { x.ane_manifest.clear(); });
    reject([](auto &x) { x.encoder_ane_manifest = "encoder.json"; });
    reject([](auto &x) { x.residency = "component_staged"; });
    reject([](auto &x) { x.width = 1024; });
    reject([](auto &x) { x.height = 1024; });
    reject([](auto &x) { x.steps = 20; });
    reject([](auto &x) { x.steps = 6; });
    reject([](auto &x) { x.loras.push_back({"adapter.safetensors", 1.f, "transformer"}); });
    reject([](auto &x) { x.prompt_enhance = true; });
    reject([](auto &x) { x.qwen21_reference_size = 512; });
    reject([](auto &x) { x.memory_constrained.enabled = true; });
    reject([](auto &x) { x.memory_budget_bytes = 8ull << 30; });
    reject([](auto &x) { x.operation = "image.transform"; });
    tc::InputAsset image;
    image.kind = "image";
    image.role = "reference";
    image.path = "image.png";
    reject([&](auto &x) { x.inputs.push_back(image); });
    r.operation = "image.edit";
    assert(!frozen_conservative_cache_request(r));
    for (int i = 0; i < 3; ++i) {
        r.inputs.push_back(image);
        assert(frozen_conservative_cache_request(r));
    }
    r.inputs.push_back(image);
    assert(!frozen_conservative_cache_request(r));
}
void test_db_prefix_snapshot_request() {
    using tc::qwen21::db_prefix_snapshot_request;
    CacheEnvironment environment;
    tc::Request r;
    r.model = "qwen-image-2.1";
    r.operation = "image.edit";
    r.qwen21_dit_cache = "conservative";
    r.steps = 40;
    r.allow_approximation = true;
    tc::InputAsset reference;
    reference.kind = "image";
    reference.role = "reference";
    reference.path = "reference.png";
    r.inputs.push_back(reference);
    const auto options = tc::qwen21::db_cache_options(r);
    assert(db_prefix_snapshot_request(r, options, true));
    assert(!db_prefix_snapshot_request(r, options, false));
    const auto valid = r;
    auto reject = [&](auto change) {
        auto other = valid;
        change(other);
        assert(!db_prefix_snapshot_request(other, options, true));
    };
    reject([](auto &x) { x.model = "other"; });
    reject([](auto &x) { x.operation = "image.generate"; });
    reject([](auto &x) { x.inputs.clear(); });
    reject([&](auto &x) { x.inputs.assign(4, reference); });
    reject([](auto &x) { x.execution = "gpu_ane"; });
    reject([](auto &x) { x.residency = "component_staged"; });
    reject([](auto &x) { x.allow_approximation = false; });
    reject([](auto &x) { x.width = 1024; });
    reject([](auto &x) { x.height = 1024; });
    reject([](auto &x) { x.steps = 19; });
    reject([](auto &x) { x.steps = 41; });
    reject([](auto &x) { x.qwen21_reference_size = 512; });
    reject([](auto &x) { x.loras.push_back({"adapter", 1.f, "transformer"}); });
    reject([](auto &x) { x.prompt_enhance = true; });
    reject([](auto &x) { x.ane_manifest = "frozen.json"; });
    reject([](auto &x) { x.encoder_ane_manifest = "encoder.json"; });
    reject([](auto &x) { x.hybrid_mlp_mode = "runtime"; });
    reject([](auto &x) { x.qwen21_denoiser_backend = "native_metal"; });
    reject([](auto &x) { x.memory_constrained.enabled = true; });
    reject([](auto &x) { x.memory_budget_bytes = 8ull << 30; });
    for (const char *mode : {"off", "balanced", "fast"}) {
        r.qwen21_dit_cache = mode;
        assert(!db_prefix_snapshot_request(r, tc::qwen21::db_cache_options(r), true));
    }
    auto diagnostic = options;
    diagnostic.diagnostic = true;
    assert(!db_prefix_snapshot_request(valid, diagnostic, true));
    auto frozen = valid;
    frozen.execution = "gpu_ane";
    frozen.qwen21_w8a8 = true;
    frozen.ane_manifest = "verified-at-load.json";
    assert(db_prefix_snapshot_request(frozen, options, true));
    assert(!db_prefix_snapshot_request(frozen, options, false));
    for (int i = 0; i < 6; ++i) {
        auto bad = frozen;
        if (i == 0) bad.steps = 39;
        if (i == 1) bad.qwen21_w8a8 = false;
        if (i == 2) bad.qwen21_gpu_w8a16 = true;
        if (i == 3) bad.qwen21_gpu_full_ffn_blocks = {3, 5, 7};
        if (i == 4) bad.hybrid_mlp_mode = "base_fused";
        if (i == 5) bad.ane_manifest.clear();
        assert(!db_prefix_snapshot_request(bad, options, true));
    }
}
void test_student_prefix_snapshot_request() {
    using tc::qwen21::student_prefix_snapshot_request;
    tc::Request r;
    r.model = "qwen-image-2.1";
    r.operation = "image.edit";
    r.steps = 6;
    r.qwen21_reference_size = 512;
    r.allow_approximation = true;
    r.lora_strategy = "inference_time";
    tc::InputAsset image;
    image.kind = "image"; image.role = "reference"; image.path = "reference.png";
    r.inputs.push_back(image);
    r.loras.push_back({std::string(tc::qwen21::viggle_v021_adapters[1].filename), 1.f, "transformer"});
    assert(student_prefix_snapshot_request(r, true));
    assert(!student_prefix_snapshot_request(r, false));
    for (const auto &adapter : tc::qwen21::viggle_v021_adapters)
        for (int size : {512, 1024}) {
            auto candidate = r;
            candidate.qwen21_reference_size = size;
            candidate.loras[0].path = adapter.filename;
            assert(student_prefix_snapshot_request(candidate, true));
        }
    auto reject = [&](auto change) {
        auto bad = r;
        change(bad);
        assert(!student_prefix_snapshot_request(bad, true));
    };
    reject([](auto &x) { x.model = "other"; });
    reject([](auto &x) { x.execution = "gpu_ane"; });
    reject([](auto &x) { x.residency = "component_staged"; });
    reject([](auto &x) { x.operation = "image.generate"; });
    reject([](auto &x) { x.width = 1024; });
    reject([](auto &x) { x.height = 1024; });
    reject([](auto &x) { x.steps = 40; });
    reject([](auto &x) { x.allow_approximation = false; });
    reject([](auto &x) { x.qwen21_reference_size = 256; });
    reject([](auto &x) { x.inputs.clear(); });
    reject([&](auto &x) { x.inputs.assign(4, image); });
    reject([](auto &x) { x.loras.clear(); });
    reject([&](auto &x) { x.loras.push_back(r.loras.front()); });
    reject([](auto &x) { x.loras[0].path = "unknown.safetensors"; });
    reject([](auto &x) { x.loras[0].role = "text_encoder"; });
    reject([](auto &x) { x.loras[0].strength = .75f; });
    reject([](auto &x) { x.lora_strategy = "in_memory_merge"; });
    reject([](auto &x) { x.qwen21_dit_cache = "conservative"; });
    reject([](auto &x) { x.prompt_enhance = true; });
    reject([](auto &x) { x.prompt_enhance_edit_experimental = true; });
    reject([](auto &x) { x.hybrid_mlp_mode = "runtime"; });
    reject([](auto &x) { x.qwen21_denoiser_backend = "native_metal"; });
    reject([](auto &x) { x.qwen21_w8a8 = true; });
    reject([](auto &x) { x.qwen21_gpu_w8a16 = true; });
    reject([](auto &x) { x.qwen21_gpu_full_ffn_blocks = {3, 5, 7}; });
    reject([](auto &x) { x.ane_manifest = "frozen.json"; });
    reject([](auto &x) { x.encoder_ane_manifest = "encoder.json"; });
    reject([](auto &x) { x.memory_constrained.enabled = true; });
    reject([](auto &x) { x.memory_budget_bytes = 8ull << 30; });
}
void test_student_five_step_request() {
    using namespace tc::qwen21;
    const char *flag = "TURBOCIDER_QWEN21_STUDENT_FIVE_STEP";
    tc::Request r;
    r.model = "qwen-image-2.1"; r.operation = "image.generate";
    r.steps = 5; r.audio = false; r.allow_approximation = true; r.lora_strategy = "inference_time";
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename), 1.f, "transformer"});
    unsetenv(flag);
    assert(student_five_step_request(r));
    assert(!configured_student_five_step_request(r));
    setenv(flag, "0", 1); assert(!configured_student_five_step_request(r));
    setenv(flag, "1", 1); assert(configured_student_five_step_request(r));
    // Six-step replay and block-observer qualification must not expand.
    assert(!pinned_student_gpu_request(r) && !student_block_observer_request(r));
    auto reject = [&](auto change) {
        auto bad = r; change(bad);
        assert(!student_five_step_request(bad));
        bool rejected = false;
        try { (void)configured_student_five_step_request(bad); }
        catch (const std::exception &) { rejected = true; }
        assert(rejected);
    };
    reject([](auto &x) { x.steps = 6; });
    reject([](auto &x) { x.steps = 4; });
    reject([](auto &x) { x.model = "other"; });
    reject([](auto &x) { x.execution = "gpu_ane"; });
    reject([](auto &x) { x.hybrid_mlp_mode = "runtime"; });
    reject([](auto &x) { x.qwen21_denoiser_backend = "native-metal-graph"; });
    reject([](auto &x) { x.residency = "component_staged"; });
    reject([](auto &x) { x.allow_approximation = false; });
    reject([](auto &x) { x.width = 1024; });
    reject([](auto &x) { x.height = 1024; });
    reject([](auto &x) { x.frames = 2; });
    reject([](auto &x) { x.audio = true; });
    reject([](auto &x) { x.prompt_enhance = true; });
    reject([](auto &x) { x.prompt_enhance_edit_experimental = true; });
    reject([](auto &x) { x.memory_constrained.enabled = true; });
    reject([](auto &x) { x.memory_budget_bytes = 1ull << 30; });
    reject([](auto &x) { x.qwen21_w8a8 = true; });
    reject([](auto &x) { x.qwen21_gpu_w8a16 = true; });
    reject([](auto &x) { x.qwen21_gpu_full_ffn_blocks = {3,5,7}; });
    reject([](auto &x) { x.ane_manifest = "frozen.json"; });
    reject([](auto &x) { x.encoder_ane_manifest = "encoder.json"; });
    reject([](auto &x) { x.qwen21_dit_cache = "conservative"; });
    reject([](auto &x) { x.lora_strategy = "in_memory_merge"; });
    reject([](auto &x) { x.loras.clear(); });
    reject([](auto &x) { x.loras.push_back(x.loras[0]); });
    reject([](auto &x) { x.loras[0].path = viggle_v021_adapters[1].filename; });
    reject([](auto &x) { x.loras[0].path = "unknown.safetensors"; });
    reject([](auto &x) { x.loras[0].role = "text_encoder"; });
    reject([](auto &x) { x.loras[0].strength = .75f; });
    reject([](auto &x) { x.operation = "image.edit"; });
    reject([](auto &x) { x.qwen21_reference_size = 512; });
    tc::InputAsset image;
    image.kind = "image"; image.role = "reference"; image.path = "ref.png";
    auto edit = r;
    edit.operation = "image.edit"; edit.qwen21_reference_size = 512;
    for (int count : {1,2,3}) {
        edit.inputs.assign(count, image);
        assert(configured_student_five_step_request(edit));
        assert(student_prefix_snapshot_request(edit, true));
        assert(!student_prefix_snapshot_request(edit, false));
        assert(!student_block_observer_request(edit));
    }
    edit.inputs.assign(4, image); assert(!student_five_step_request(edit));
    for (const char *bad : {"", "true", "01", "5", " 1", "1 "}) {
        setenv(flag, bad, 1); bool rejected = false;
        try { (void)configured_student_five_step_request(r); }
        catch (const std::exception &) { rejected = true; }
        assert(rejected);
    }
    setenv(flag, "1", 1);
    for (const char *conflict : {"TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
                                  "TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION",
                                  "TURBOCIDER_QWEN21_VAE_FUSED_NORM"}) {
        setenv(conflict, "16", 1); bool rejected = false;
        try { (void)configured_student_five_step_request(r); }
        catch (const std::exception &) { rejected = true; }
        assert(rejected); unsetenv(conflict);
    }
    // Explicit sampler recipe must partition cross-tier snapshot identities.
    assert(viggle_v021_sampler_identity(5) != viggle_v021_sampler_identity(6));
    assert(viggle_v021_sampler_identity(4).empty());
    unsetenv(flag);
}
void test_tokenizer_reuse_request() {
    using tc::qwen21::configured_tokenizer_reuse_request;
    const char *flag="TURBOCIDER_QWEN21_TOKENIZER_REUSE";
    tc::Request r;r.model="qwen-image-2.1";r.operation="image.generate";r.audio=false;
    unsetenv(flag);assert(!configured_tokenizer_reuse_request(r));
    setenv(flag,"0",1);assert(!configured_tokenizer_reuse_request(r));
    setenv(flag,"1",1);assert(configured_tokenizer_reuse_request(r));
    // This is not an algorithm approximation; base exact requests are valid.
    assert(!r.allow_approximation);
    auto reject=[&](auto change){auto bad=r;change(bad);bool rejected=false;
        try{(void)configured_tokenizer_reuse_request(bad);}catch(const std::exception&){rejected=true;}
        assert(rejected);
    };
    reject([](auto &x){x.model="other";});reject([](auto &x){x.execution="gpu_ane";});
    reject([](auto &x){x.width=1024;});reject([](auto &x){x.height=768;});
    reject([](auto &x){x.residency="component_staged";});
    reject([](auto &x){x.qwen21_denoiser_backend="native-metal-graph";});
    reject([](auto &x){x.hybrid_mlp_mode="runtime";});
    reject([](auto &x){x.prompt_enhance=true;});reject([](auto &x){x.prompt_enhance_edit_experimental=true;});
    reject([](auto &x){x.ane_manifest="ane.json";});reject([](auto &x){x.encoder_ane_manifest="encoder.json";});
    reject([](auto &x){x.qwen21_w8a8=true;});reject([](auto &x){x.qwen21_gpu_w8a16=true;});
    reject([](auto &x){x.memory_budget_bytes=8ull<<30;});reject([](auto &x){x.memory_constrained.enabled=true;});
    reject([](auto &x){x.operation="image.edit";});
    tc::InputAsset image;image.kind="image";image.role="reference";image.path="ref.png";
    auto edit=r;edit.operation="image.edit";
    for(int count:{1,2,3})for(int size:{512,1024}) {
        edit.inputs.assign(count,image);edit.qwen21_reference_size=size;
        assert(configured_tokenizer_reuse_request(edit));
    }
    edit.inputs.assign(4,image);bool rejected=false;
    try{(void)configured_tokenizer_reuse_request(edit);}catch(const std::exception&){rejected=true;}assert(rejected);
    for(const char *invalid:{"","true","01"," 1","2"}) {
        setenv(flag,invalid,1);rejected=false;
        try{(void)configured_tokenizer_reuse_request(r);}catch(const std::exception&){rejected=true;}assert(rejected);
    }
    setenv(flag,"1",1);
    setenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU","1",1);rejected=false;
    try{(void)configured_tokenizer_reuse_request(r);}catch(const std::exception&){rejected=true;}assert(rejected);
    unsetenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU");unsetenv(flag);
}
void test_student_graph_request() {
    using namespace tc::qwen21;
    tc::Request r;
    r.model = "qwen-image-2.1";
    const char *flag = "TURBOCIDER_QWEN21_RETAIN_STUDENT_GRAPH";
    const char *previous = std::getenv(flag);
    const bool had_previous = previous != nullptr;
    const std::string previous_value = previous ? previous : "";
    assert(unsetenv(flag) == 0);
    assert(!configured_student_graph_request(r));
    assert(setenv(flag, "0", 1) == 0);
    assert(!configured_student_graph_request(r));
    assert(setenv(flag, "1", 1) == 0);
    bool retired = false;
    try { (void)configured_student_graph_request(r); }
    catch (const std::exception &error) {
        retired = std::string_view(error.what()).find("retired: no measured net benefit") != std::string_view::npos;
    }
    assert(retired);
    for (const char *invalid : {"", "bad", "true", "01", " 1", "2"}) {
        assert(setenv(flag, invalid, 1) == 0);
        bool rejected = false;
        try { (void)configured_student_graph_request(r); }
        catch (const std::exception &error) {
            rejected = std::string_view(error.what()).find("requires 0 or 1") != std::string_view::npos;
        }
        assert(rejected);
    }
    if (had_previous) assert(setenv(flag, previous_value.c_str(), 1) == 0);
    else assert(unsetenv(flag) == 0);
}
void test_student_block_observer_request() {
    using namespace tc::qwen21;
    tc::Request r;
    r.model="qwen-image-2.1"; r.steps=6; r.allow_approximation=true;
    r.lora_strategy="inference_time";
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename),1.f,"transformer"});
    assert(student_block_observer_request(r));
    tc::InputAsset ref;ref.kind="image";ref.role="reference";ref.path="ref.png";
    auto edit=r;edit.operation="image.edit";edit.inputs={ref};edit.qwen21_reference_size=512;
    assert(student_block_observer_request(edit));
    for(int i=0;i<17;++i) {
        auto invalid=edit;
        if(i==0)invalid.steps=40;if(i==1)invalid.residency="component_staged";
        if(i==2)invalid.execution="gpu_ane";if(i==3)invalid.loras[0].strength=.5f;
        if(i==4)invalid.loras[0].path=viggle_v021_adapters[1].filename;
        if(i==5)invalid.loras[0].path="unknown.safetensors";if(i==6)invalid.inputs.push_back(ref);
        if(i==7)invalid.qwen21_reference_size=1024;if(i==8)invalid.dump="trace";
        if(i==9)invalid.noise_path="noise.safetensors";if(i==10)invalid.allow_approximation=false;
        if(i==11)invalid.hybrid_mlp_mode="runtime";if(i==12)invalid.prompt_enhance=true;
        if(i==13)invalid.qwen21_dit_cache="conservative";if(i==14)invalid.memory_budget_bytes=1;
        if(i==15)invalid.operation="image.generate";if(i==16)invalid.qwen21_denoiser_backend="native_metal";
        assert(!student_block_observer_request(invalid));
    }
}
void test_fp16_vae_decode_request() {
    using namespace tc::qwen21;
    tc::Request r;r.model="qwen-image-2.1";r.allow_approximation=true;
    const char *flag="TURBOCIDER_QWEN21_VAE_FP16_CONV";
    unsetenv(flag);assert(!configured_fp16_vae_decode_request(r));
    setenv(flag,"0",1);assert(!configured_fp16_vae_decode_request(r));
    setenv(flag,"1",1);assert(configured_fp16_vae_decode_request(r));
    tc::InputAsset ref;ref.kind="image";ref.role="reference";ref.path="ref.png";
    auto edit=r;edit.operation="image.edit";edit.inputs={ref};
    assert(configured_fp16_vae_decode_request(edit));
    for(int i=0;i<13;++i) {
        auto bad=edit;
        if(i==0)bad.execution="gpu_ane";if(i==1)bad.residency="component_staged";
        if(i==2)bad.allow_approximation=false;if(i==3)bad.width=1024;
        if(i==4)bad.inputs.assign(4,ref);if(i==5)bad.hybrid_mlp_mode="runtime";
        if(i==6)bad.prompt_enhance=true;if(i==7)bad.memory_budget_bytes=1;
        if(i==8)bad.ane_manifest="manifest";if(i==9)bad.encoder_ane_manifest="manifest";
        if(i==10)bad.qwen21_w8a8=true;if(i==11)bad.qwen21_denoiser_backend="native_metal";
        if(i==12)bad.operation="video.generate";
        bool rejected=false;try{(void)configured_fp16_vae_decode_request(bad);}catch(const std::exception&){rejected=true;}
        assert(rejected);
    }
    for(const char *value:{"", "bad", "true", "01", "2"}) {
        setenv(flag,value,1);bool rejected=false;
        try{(void)configured_fp16_vae_decode_request(r);}catch(const std::exception&){rejected=true;}
        assert(rejected);
    }
    unsetenv(flag);
}
void test_fused_vae_norm_request() {
    using namespace tc::qwen21;
    tc::Request r;r.model="qwen-image-2.1";r.allow_approximation=true;
    const char *flag="TURBOCIDER_QWEN21_VAE_FUSED_NORM";
    unsetenv(flag);assert(!configured_fused_vae_norm_request(r));
    setenv(flag,"0",1);assert(!configured_fused_vae_norm_request(r));
    setenv(flag,"1",1);assert(configured_fused_vae_norm_request(r));
    auto bad=r;bad.allow_approximation=false;bool rejected=false;
    try{(void)configured_fused_vae_norm_request(bad);}catch(const std::exception&){rejected=true;}
    assert(rejected);
    for(const char *value:{"", "bad", "true", "01", "2"}) {
        setenv(flag,value,1);rejected=false;
        try{(void)configured_fused_vae_norm_request(r);}catch(const std::exception&){rejected=true;}
        assert(rejected);
    }
    unsetenv(flag);
}
} // namespace

void test_corrected_qkv_request() {
    using namespace tc::qwen21;tc::Request r;r.model="qwen-image-2.1";r.operation="image.generate";
    r.steps=6;r.audio=false;r.allow_approximation=true;r.lora_strategy="inference_time";
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename),1.f,"transformer"});
    unsetenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT");
    unsetenv("TURBOCIDER_QWEN21_CORRECTED_QKV");assert(!configured_corrected_qkv_request(r));
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","1",1);setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);
    auto rejects=[&](const auto &bad){bool rejected=false;try{(void)configured_corrected_qkv_request(bad);}catch(const std::exception&){rejected=true;}assert(rejected);};
    assert(configured_corrected_qkv_request(r));
    for(int i=0;i<18;++i){auto bad=r;
        if(i==0)bad.steps=5;if(i==1)bad.width=1024;if(i==2)bad.allow_approximation=false;if(i==3)bad.execution="gpu_ane";
        if(i==4)bad.loras[0].strength=.5f;if(i==5)bad.lora_strategy="in_memory_merge";if(i==6)bad.loras.clear();
        if(i==7)bad.qwen21_dit_cache="conservative";if(i==8)bad.residency="component_staged";
        if(i==9)bad.loras[0].path=viggle_v021_adapters[1].filename;if(i==10)bad.inputs.resize(1);
        if(i==11)bad.operation="image.edit";if(i==12)bad.audio=true;if(i==13)bad.frames=2;
        if(i==14)bad.prompt_enhance=true;if(i==15)bad.qwen21_denoiser_backend="native_metal";
        if(i==16)bad.memory_budget_bytes=1;if(i==17)bad.hybrid_mlp_mode="runtime";rejects(bad);
    }
    r.operation="image.edit";r.qwen21_reference_size=512;r.inputs.resize(1);assert(configured_corrected_qkv_request(r));
    auto bad=r;bad.inputs.resize(2);rejects(bad);bad=r;bad.qwen21_reference_size=1024;rejects(bad);
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","1",1);assert(configured_corrected_qkv_request(r));
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","0",1);rejects(r);
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","1",1);
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","true",1);rejects(r);
    unsetenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT");
    for(const char *flag:{"TURBOCIDER_QWEN21_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_METAL_QK_ROPE","TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE",
        "TURBOCIDER_QWEN21_BF16_STREAMING","TURBOCIDER_QWEN21_VAE_FP16_CONV","TURBOCIDER_QWEN21_TOKENIZER_REUSE"}){
        setenv(flag,"1",1);rejects(r);unsetenv(flag);
    }
    unsetenv("TURBOCIDER_QWEN21_LORA_BF16_AB");rejects(r);
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","bogus",1);rejects(r);unsetenv("TURBOCIDER_QWEN21_CORRECTED_QKV");
}

void test_five_step_direct_qkv_composition() {
    using namespace tc::qwen21;tc::Request r;r.model="qwen-image-2.1";r.operation="image.generate";
    r.steps=5;r.audio=false;r.allow_approximation=true;r.lora_strategy="inference_time";
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename),1.f,"transformer"});
    for(const char *flag:{"TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","TURBOCIDER_QWEN21_CORRECTED_QKV",
        "TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","TURBOCIDER_QWEN21_LORA_BF16_AB"})setenv(flag,"1",1);
    assert(configured_student_five_step_request(r)&&configured_corrected_qkv_request(r));
    auto reject=[&](const auto &bad){bool denied=false;try{(void)configured_corrected_qkv_request(bad);}catch(const std::exception&){denied=true;}assert(denied);};
    assert(!student_block_observer_request(r));assert(!pinned_student_gpu_request(r));
    auto bad=r;bad.steps=6;reject(bad);bad=r;bad.steps=4;reject(bad);
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","0",1);reject(r);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","1",1);
    setenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","0",1);reject(r);setenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","1",1);
    r.operation="image.edit";r.inputs.resize(1);r.qwen21_reference_size=512;assert(configured_corrected_qkv_request(r));
    bad=r;bad.inputs.resize(2);reject(bad);bad=r;bad.qwen21_reference_size=1024;reject(bad);
    for(const char *flag:{"TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","TURBOCIDER_QWEN21_CORRECTED_QKV",
        "TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","TURBOCIDER_QWEN21_LORA_BF16_AB"})unsetenv(flag);
}

void test_direct_qkv_prefix_composition() {
    using namespace tc::qwen21;tc::Request r;r.model="qwen-image-2.1";r.operation="image.edit";
    r.steps=6;r.audio=false;r.allow_approximation=true;r.lora_strategy="inference_time";
    r.qwen21_reference_size=512;r.inputs.resize(1);
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename),1.f,"transformer"});
    const char *flags[]={"TURBOCIDER_QWEN21_CORRECTED_QKV","TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT",
        "TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX","TURBOCIDER_QWEN21_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_LORA_BF16_AB"};
    for(const auto *flag:flags)setenv(flag,"1",1);
    assert(configured_corrected_qkv_request(r));
    auto rejects=[&](const auto &bad){bool denied=false;try{(void)configured_corrected_qkv_request(bad);}catch(const std::exception&){denied=true;}assert(denied);};
    for(const auto *flag:flags){setenv(flag,"0",1);rejects(r);setenv(flag,"1",1);}
    for(const char *value:{"true","-1",""}){setenv(flags[2],value,1);rejects(r);}
    setenv(flags[2],"1",1);
    for(int i=0;i<6;++i){auto bad=r;
        if(i==0){bad.operation="image.generate";bad.inputs.clear();bad.qwen21_reference_size=1024;}
        if(i==1)bad.inputs.resize(2);if(i==2)bad.qwen21_reference_size=1024;
        if(i==3)bad.dump="raw";if(i==4)bad.steps=5;if(i==5)bad.execution="gpu_ane";rejects(bad);
    }
    r.steps=5;setenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","1",1);
    assert(configured_corrected_qkv_request(r));assert(!student_block_observer_request(r));
    unsetenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP");for(const auto *flag:flags)unsetenv(flag);
}

void test_lora_swiglu_request() {
    using namespace tc::qwen21;tc::Request r;r.model="qwen-image-2.1";r.operation="image.generate";
    r.steps=6;r.audio=false;r.allow_approximation=true;r.lora_strategy="inference_time";
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename),1.f,"transformer"});
    const char *flag="TURBOCIDER_QWEN21_LORA_SWIGLU";unsetenv(flag);assert(!configured_lora_swiglu_request(r));
    setenv(flag,"1",1);setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);
    auto rejects=[&](const auto &bad){bool denied=false;try{(void)configured_lora_swiglu_request(bad);}catch(const std::exception&){denied=true;}assert(denied);};
    assert(configured_lora_swiglu_request(r));
    for(int i=0;i<15;++i){auto bad=r;
        if(i==0)bad.steps=5;if(i==1)bad.width=1024;if(i==2)bad.allow_approximation=false;if(i==3)bad.execution="gpu_ane";
        if(i==4)bad.loras[0].strength=.5f;if(i==5)bad.lora_strategy="in_memory_merge";if(i==6)bad.loras.clear();
        if(i==7)bad.qwen21_dit_cache="conservative";if(i==8)bad.residency="component_staged";if(i==9)bad.inputs.resize(1);
        if(i==10)bad.loras[0].path=viggle_v021_adapters[1].filename;if(i==11)bad.audio=true;if(i==12)bad.frames=2;
        if(i==13)bad.memory_budget_bytes=1;if(i==14)bad.qwen21_denoiser_backend="native_metal";rejects(bad);
    }
    for(const char *value:{"", "true","01","2"}){setenv(flag,value,1);rejects(r);}setenv(flag,"1",1);
    for(const char *other:{"TURBOCIDER_QWEN21_VAE_FP16_CONV","TURBOCIDER_QWEN21_VAE_FUSED_NORM","TURBOCIDER_QWEN21_TOKENIZER_REUSE",
        "TURBOCIDER_QWEN21_PROFILE_GPU_BLOCKS","TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN","TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC"}){
        setenv(other,"1",1);rejects(r);unsetenv(other);}
    setenv("TURBOCIDER_QWEN21_ENCODER_BACKEND","ane",1);rejects(r);unsetenv("TURBOCIDER_QWEN21_ENCODER_BACKEND");
    setenv("TURBOCIDER_QWEN21_CORRECTED_QKV","1",1);rejects(r);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","1",1);assert(configured_lora_swiglu_request(r));
    r.operation="image.edit";r.qwen21_reference_size=512;r.inputs.resize(1);
    for(int qkv:{0,1})for(int prefix:{0,1}){
        setenv("TURBOCIDER_QWEN21_CORRECTED_QKV",qkv?"1":"0",1);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT",qkv?"1":"0",1);
        setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX",qkv&&prefix?"1":"0",1);setenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT",prefix?"1":"0",1);
        assert(configured_lora_swiglu_request(r));
    }
    auto bad=r;bad.inputs.resize(2);rejects(bad);bad=r;bad.dump="raw";rejects(bad);
    r.steps=5;setenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","1",1);assert(configured_lora_swiglu_request(r));
    assert(!pinned_student_gpu_request(r)&&!student_block_observer_request(r));
    for(const auto *name:{flag,"TURBOCIDER_QWEN21_LORA_BF16_AB","TURBOCIDER_QWEN21_CORRECTED_QKV","TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT",
        "TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX","TURBOCIDER_QWEN21_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_STUDENT_FIVE_STEP"})unsetenv(name);
}

void test_bounded_async_step_pair_request() {
    using namespace tc::qwen21;tc::Request r;r.model="qwen-image-2.1";r.operation="image.generate";
    r.steps=6;r.audio=false;r.allow_approximation=true;r.lora_strategy="inference_time";
    r.loras.push_back({std::string(viggle_v021_adapters[0].filename),1.f,"transformer"});
    const char *flag="TURBOCIDER_QWEN21_ASYNC_STEP_PAIR";unsetenv(flag);assert(!configured_async_step_pair_request(r));
    setenv(flag,"1",1);setenv("TURBOCIDER_QWEN21_LORA_BF16_AB","1",1);assert(configured_async_step_pair_request(r));
    auto reject=[&](const auto &bad){bool denied=false;try{(void)configured_async_step_pair_request(bad);}catch(const std::exception&){denied=true;}assert(denied);};
    for(int i=0;i<12;++i){auto bad=r;if(i==0)bad.steps=5;if(i==1)bad.steps=40;if(i==2)bad.width=1024;if(i==3)bad.allow_approximation=false;
        if(i==4)bad.execution="gpu_ane";if(i==5)bad.residency="component_staged";if(i==6)bad.loras.clear();if(i==7)bad.loras[0].strength=.5f;
        if(i==8)bad.qwen21_dit_cache="conservative";if(i==9)bad.memory_budget_bytes=1;if(i==10)bad.loras[0].path=viggle_v021_adapters[1].filename;
        if(i==11)bad.qwen21_denoiser_backend="native_metal";reject(bad);}
    for(const char *value:{"true","01","2",""}){setenv(flag,value,1);reject(r);}setenv(flag,"1",1);
    for(const char *other:{"TURBOCIDER_QWEN21_PROFILE_STEPS","TURBOCIDER_ANE_CALIBRATION_REPLAY","TURBOCIDER_QWEN21_TOKENIZER_REUSE","TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN"}){
        setenv(other,"1",1);reject(r);unsetenv(other);}
    setenv("TURBOCIDER_ANE_CALIBRATION_DIR","capture",1);reject(r);unsetenv("TURBOCIDER_ANE_CALIBRATION_DIR");
    r.operation="image.edit";r.inputs.resize(1);r.qwen21_reference_size=512;
    for(int fused:{0,1})for(int direct:{0,1})for(int prefix:{0,1}){
        setenv("TURBOCIDER_QWEN21_LORA_SWIGLU",fused?"1":"0",1);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV",direct?"1":"0",1);
        setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT",direct?"1":"0",1);setenv("TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX",direct&&prefix?"1":"0",1);
        setenv("TURBOCIDER_QWEN21_PREFIX_SNAPSHOT",prefix?"1":"0",1);assert(configured_async_step_pair_request(r));
    }
    r.steps=5;setenv("TURBOCIDER_QWEN21_STUDENT_FIVE_STEP","1",1);assert(configured_async_step_pair_request(r));
    assert(!pinned_student_gpu_request(r)&&!student_block_observer_request(r));
    for(const auto *name:{flag,"TURBOCIDER_QWEN21_LORA_BF16_AB","TURBOCIDER_QWEN21_LORA_SWIGLU","TURBOCIDER_QWEN21_CORRECTED_QKV",
        "TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT","TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX","TURBOCIDER_QWEN21_PREFIX_SNAPSHOT","TURBOCIDER_QWEN21_STUDENT_FIVE_STEP"})unsetenv(name);
}

int main() {
    test_bounded_async_step_pair_request();
    test_lora_swiglu_request();
    test_direct_qkv_prefix_composition();
    test_five_step_direct_qkv_composition();
    test_corrected_qkv_request();
    using namespace tc::qwen21;
    test_cache_options();
    test_frozen_conservative_cache_request();
    test_db_prefix_snapshot_request();
    test_student_prefix_snapshot_request();
    test_student_five_step_request();
    test_tokenizer_reuse_request();
    test_student_graph_request();
    test_student_block_observer_request();
    test_fp16_vae_decode_request();
    test_fused_vae_norm_request();
    assert(gguf_read_buffer_mib(nullptr)==1 && gguf_read_buffer_mib("1")==1 &&
        gguf_read_buffer_mib("4")==4 && gguf_read_buffer_mib("8")==8);
    for(const char *value:{"","0","2","16","08"," 8","8 ","8.0","eight"})
        assert(gguf_read_buffer_mib(value)==0);
    unsetenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS");
    assert(gguf_fast_defaults_enabled());
    setenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS","0",1);assert(!gguf_fast_defaults_enabled());
    setenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS","1",1);assert(gguf_fast_defaults_enabled());
    unsetenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS");
    assert(option_or_default(nullptr,true) && !option_or_default(nullptr,false));
    assert(!option_or_default("0",true) && option_or_default("1",false));
    auto rejected=[](auto action) {
        bool threw=false;
        try { action(); } catch(const std::exception &) { threw=true; }
        assert(threw);
    };
    for(const char *bad:{"","true","01"," 1","2"}) {
        setenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS",bad,1);
        rejected([] { (void)gguf_fast_defaults_enabled(); });
        rejected([&] { (void)option_or_default(bad,true); });
    }
    unsetenv("TURBOCIDER_QWEN21_GGUF_FAST_DEFAULTS");
    tc::Request base;base.model="qwen-image-2.1";base.operation="image.generate";base.execution="gpu";
    base.width=base.height=512;base.residency="resident";base.allow_approximation=true;
    assert(gguf_fast_encoder_request(base,true,true));
    assert(!gguf_fast_encoder_request(base,false,true) && !gguf_fast_encoder_request(base,true,false));
    for(int mode=0;mode<16;++mode) {
        auto invalid=base;
        if(mode==0)invalid.model="other";
        if(mode==1)invalid.width=1024;
        if(mode==2)invalid.operation="image.edit";
        if(mode==3)invalid.inputs.resize(1);
        if(mode==4)invalid.loras.push_back({"adapter",1.f,"transformer"});
        if(mode==5)invalid.residency="component_staged";
        if(mode==6)invalid.allow_approximation=false;
        if(mode==7)invalid.prompt_enhance=true;
        if(mode==8)invalid.memory_budget_bytes=1;
        if(mode==9)invalid.qwen21_w8a8=true;
        if(mode==10)invalid.execution="cpu";
        if(mode==11)invalid.encoder_ane_manifest="encoder.json";
        if(mode==12)invalid.qwen21_gpu_w8a16=true;
        if(mode==13)invalid.memory_constrained.enabled=true;
        if(mode==14)invalid.streaming.enabled=true;
        if(mode==15)invalid.height=1024;
        assert(!gguf_fast_encoder_request(invalid,true,true));
    }
    auto mixed=base;mixed.execution="gpu_ane";mixed.hybrid_mlp_mode="runtime";
    assert(gguf_fast_encoder_request(mixed,true,true));
    mixed.hybrid_mlp_mode="auto";assert(!gguf_fast_encoder_request(mixed,true,true));
    unsetenv("TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU");
    unsetenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU");
    assert(!compiled_encoder_gpu(base) && compiled_encoder_gpu(base,true));
    setenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU","0",1);
    assert(!compiled_encoder_gpu(base,true));
    setenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU","1",1);
    assert(compiled_encoder_gpu(base));
    auto invalid=base;invalid.width=1024;
    rejected([&] { (void)compiled_encoder_gpu(invalid); });
    unsetenv("TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU");
    assert(!option_enabled(nullptr));
    assert(!option_enabled("0"));
    assert(!option_enabled(""));
    assert(!option_enabled("true"));
    assert(option_enabled("1"));
    assert(binary_option_or_unset(nullptr));
    assert(binary_option_or_unset("0"));
    assert(binary_option_or_unset("1"));
    assert(!binary_option_or_unset("true"));
    assert(runtime_ffn_phase(nullptr)==RuntimeFfnPhase::All);
    assert(runtime_ffn_phase("all")==RuntimeFfnPhase::All);
    assert(runtime_ffn_phase("prefill")==RuntimeFfnPhase::Prefill);
    assert(runtime_ffn_phase("decode")==RuntimeFfnPhase::Decode);
    for(const char *bad:{"","0","1","ALL","gpu","prefill,decode"})
        assert(runtime_ffn_phase(bad)==RuntimeFfnPhase::Invalid);
    assert(runtime_ffn_phase_identity(RuntimeFfnPhase::All).empty());
    assert(runtime_ffn_phase_identity(RuntimeFfnPhase::Prefill)!=runtime_ffn_phase_identity(RuntimeFfnPhase::Decode));
    for(auto phase:{RuntimeFfnPhase::All,RuntimeFfnPhase::Prefill,RuntimeFfnPhase::Decode}) {
        assert(runtime_ffn_phase_runs(phase,false)==(phase!=RuntimeFfnPhase::Decode));
        assert(runtime_ffn_phase_runs(phase,true)==(phase!=RuntimeFfnPhase::Prefill));
    }
    assert(tiled_prefill_layer_count("0") == 0);
    assert(tiled_prefill_layer_count("1") == 32);
    assert(tiled_prefill_layer_count("16") == 16);
    assert(tiled_prefill_layer_count("17") == -1);
    assert(db_cache_threshold(nullptr) == 0.08f);
    assert(db_cache_threshold("0.25") == 0.25f);
    assert(db_cache_threshold("nan") < 0);
    assert(db_cache_threshold("0.51") < 0);
    assert(db_cache_max_consecutive(nullptr) == 2);
    assert(db_cache_max_consecutive("1") == 1);
    assert(db_cache_max_consecutive("4") == 4);
    assert(db_cache_max_consecutive("8") == 8);
    assert(db_cache_max_consecutive("0") < 0);
    assert(db_cache_max_consecutive("9") < 0);
    assert(db_cache_max_consecutive("4oops") < 0);
    assert(db_cache_front_blocks(nullptr) == 8);
    assert(db_cache_front_blocks("1") == 1);
    assert(db_cache_front_blocks("8") == 8);
    assert(db_cache_front_blocks("2") < 0);
    assert(db_cache_front_blocks("01") < 0);
    assert(db_cache_warmup_steps(nullptr) == 8);
    assert(db_cache_warmup_steps("4") == 4);
    assert(db_cache_warmup_steps("8") == 8);
    assert(db_cache_warmup_steps("0") < 0);
    assert(db_cache_warmup_steps("4oops") < 0);

    tc::Request lora;
    lora.model="qwen-image-2.1";
    lora.width=lora.height=1024;lora.steps=6;lora.allow_approximation=true;
    lora.loras.push_back({"adapter.safetensors",1.f,"transformer"});
    unsetenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC");
    assert(!lora_1024_generation(lora));
    setenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC","1",1);
    assert(lora_1024_generation(lora));
    lora.execution="gpu_ane";lora.hybrid_mlp_mode="runtime";
    assert(lora_1024_generation(lora));
    for(int mode=0;mode<11;++mode) {
        auto invalid=lora;
        if(mode==0)invalid.width=512;
        if(mode==1)invalid.steps=5;
        if(mode==2)invalid.allow_approximation=false;
        if(mode==3)invalid.operation="image.edit";
        if(mode==4)invalid.inputs.resize(1);
        if(mode==5)invalid.loras.clear();
        if(mode==6)invalid.residency="component_staged";
        if(mode==7)invalid.qwen21_w8a8=true;
        if(mode==8)invalid.hybrid_mlp_mode="lora_fused";
        if(mode==9)invalid.prompt_enhance=true;
        if(mode==10)invalid.qwen21_gpu_full_ffn_blocks={3,5,7};
        assert(!lora_1024_generation(invalid));
    }
    setenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC","0",1);
    assert(!lora_1024_generation(lora));
    unsetenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC");

    W8A8CallBudget budget{
        .steps = 20, .decode_layers = 32, .db_skipped_layers = 24,
        .total_tokens = 4096, .tile_rows = 1024,
    };
    assert(expected_w8a8_calls(budget) == 608);
    budget.db_cached_steps = 8;
    assert(expected_w8a8_calls(budget) == 416);
    budget.steps = 40;
    budget.db_cached_steps = 21;
    assert(expected_w8a8_calls(budget) == 744);

    budget.steps = 5;
    budget.db_cached_steps = 0;
    budget.final_reuse_layers = 16;
    budget.penultimate_reuse_layers = 16;
    assert(expected_w8a8_calls(budget) == 96);
    budget.final_reuse_layers = budget.penultimate_reuse_layers = 0;
    budget.tiled_prefill_layers = 16;
    assert(expected_w8a8_calls(budget) == 192);
    budget.last_target_only = true;
    assert(expected_w8a8_calls(budget) == 189);

    budget.prefix_hit = budget.tiled_prefix_reuse = true;
    budget.prefix_tokens = 2100; // incomplete prefix tile
    assert(expected_w8a8_calls(budget) == 159);
    budget.prefix_target_only = true;
    assert(expected_w8a8_calls(budget) == 144);
    budget.prefix_target_only = false;
    budget.prefix_tokens = 2048; // aligned prefix, no tile tail
    assert(expected_w8a8_calls(budget) == 144);

    budget.tiled_prefill_layers = 0;
    budget.decode_layers = 29; // 3/5/7 full GPU fallback
    assert(expected_w8a8_calls(budget) == 116);
    budget.decode_layers = 32;
    budget.decode_tiles = 2; // 1536-row rectangle on a 1024-row graph
    assert(expected_w8a8_calls(budget) == 256);
}
