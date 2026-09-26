#include "pe_generation.hpp"
#include "conditioning.hpp"
#include <algorithm>

namespace tc::qwen21::pe {
namespace {
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
TextGenerationResult sample_continuation(const Weights &weights, const Tokenizer &tokenizer,
    LanguageModel &model, Tensor hidden, TextGenerationResult result,
    const SamplingProfile &sampling, uint64_t seed, const Event &event,
    std::atomic<bool> &cancelled, int image_next_position = -1) {
    std::mt19937_64 rng(seed);
    std::unordered_set<int> generated;
    std::vector<int> answer;
    answer.reserve(sampling.max_new_tokens);
    constexpr int eos = 248044; // official PE-T2I and PE-I2I generation_config.json
    const auto decode_start = Clock::now();
    for (int step = 0; step < sampling.max_new_tokens; ++step) {
        checkpoint(cancelled);
        auto logits = mx::astype(linear(slice_axis(hidden, 1, hidden.shape(1) - 1,
            hidden.shape(1)), weights, "lm_head"), mx::float32);
        mx::eval(logits);
        require(logits.ndim() == 3 && logits.shape(0) == 1 && logits.shape(1) == 1,
                "PE sampling requires last-token logits");
        std::vector<float> values(logits.data<float>(), logits.data<float>() + logits.size());
        const int id = draw_token(token_distribution(values, generated, sampling), rng);
        answer.push_back(id);
        generated.insert(id);
        if (id == eos) { result.stopped_eos = true; break; }
        if (event && (step + 1) % 32 == 0) event("pe_decode", step + 1, sampling.max_new_tokens);
        if (step + 1 < sampling.max_new_tokens) {
            auto ids = Tensor(&id, {1, 1}, mx::int32);
            if (image_next_position < 0) hidden = model.forward_ids(ids, {}, cancelled);
            else hidden = model.forward_embeddings(
                mx::take(weights.at("model.language_model.embed_tokens.weight"), ids, 0),
                mx::full({3, 1}, image_next_position + step, mx::int32), {}, cancelled);
        }
    }
    result.decode_seconds = elapsed(decode_start);
    result.generated_tokens = int(answer.size());
    if (event) event("pe_decode", result.generated_tokens, sampling.max_new_tokens);
    checkpoint(cancelled);
    result.cached_tokens = model.cached_tokens();
    require(result.cached_tokens == result.prompt_tokens + result.generated_tokens - 1,
            "PE generation cache count mismatch");
    result.raw = tokenizer.decode(answer);
    result.generated_ids = std::move(answer);
    // The official chat template opens thinking before the first output token.
    result.rewrite = parse_prompt_rewrite_answer("<think>\n" + result.raw, image_next_position >= 0);
    return result;
}
}

TextGenerationResult generate_text(const Weights &weights, const Tokenizer &tokenizer,
    const std::string &system_prompt, const std::string &user_prompt,
    const SamplingProfile &sampling, uint64_t seed, const Event &event,
    std::atomic<bool> &cancelled, LanguageConfig language) {
    checkpoint(cancelled);
    require(sampling.max_new_tokens > 0 && sampling.max_new_tokens <= 16256,
            "invalid PE-T2I token budget");
    token_distribution({0.f}, {}, sampling);
    const auto tokens = tokenizer.raw_bounded(
        prompt_rewrite_chat(system_prompt, user_prompt), 32768 - sampling.max_new_tokens);
    TextGenerationResult result;
    result.chunked_prefill = language.chunked_prefill;
    result.prompt_tokens = tokens.valid;
    result.input_prompt_tokens = tokens.valid;
    result.next_position = tokens.valid;
    LanguageModel model(weights, std::move(language));
    auto ids = Tensor(tokens.ids.data(), {1, int(tokens.ids.size())}, mx::int32);
    Tensor hidden(0.f);
    const auto prefill_start = Clock::now();
    for (int begin = 0; begin < tokens.valid; begin += 128) {
        checkpoint(cancelled);
        const int end = std::min(begin + 128, tokens.valid);
        hidden = model.forward_ids(slice_axis(ids, 1, begin, end), {}, cancelled);
        if (event) event("pe_prefill", end, tokens.valid);
    }
    result.prefill_seconds = elapsed(prefill_start);
    return sample_continuation(weights, tokenizer, model, hidden, result, sampling, seed, event, cancelled);
}

TextGenerationResult generate_edit_features(const Weights &weights, const Tokenizer &tokenizer,
    const std::string &system_prompt, const std::string &user_prompt,
    const std::vector<ImageGrid> &grids, const std::vector<Tensor> &features,
    const SamplingProfile &sampling, uint64_t seed, const Event &event,
    std::atomic<bool> &cancelled, LanguageConfig language) {
    checkpoint(cancelled);
    require(sampling.max_new_tokens > 0 && sampling.max_new_tokens <= 24000, "invalid PE-I2I token budget");
    token_distribution({0.f}, {}, sampling);
    require(!grids.empty() && grids.size() <= 10 && features.size() == grids.size(),
            "PE-I2I requires 1...10 matching grids/features");
    auto tokens = tokenizer.raw_bounded(prompt_rewrite_chat(system_prompt, user_prompt, grids.size()),
                                        32768 - sampling.max_new_tokens);
    auto layout = image_layout(tokens, grids, 32768 - sampling.max_new_tokens);
    for (size_t i = 0; i < features.size(); ++i) {
        checkpoint(cancelled);
        require(features[i].shape() == mx::Shape{layout.images[i].count, language.hidden},
                "PE-I2I feature/grid shape mismatch");
        require(features[i].dtype() == mx::float32 || features[i].dtype() == mx::bfloat16 ||
                    features[i].dtype() == mx::float16, "PE-I2I features must be floating point");
        require(mx::all(mx::isfinite(features[i])).item<bool>(), "nonfinite PE-I2I features");
    }
    auto embeddings = image_embeddings(layout, weights.at("model.language_model.embed_tokens.weight"), features);
    auto positions = layout.position_tensor();
    TextGenerationResult result;
    result.chunked_prefill = language.chunked_prefill;
    result.prompt_tokens = int(layout.ids.size());
    result.input_prompt_tokens = tokens.valid;
    result.next_position = layout.next_position;
    LanguageModel model(weights, std::move(language));
    Tensor hidden(0.f);
    const auto prefill_start = Clock::now();
    for (int begin = 0; begin < result.prompt_tokens; begin += 128) {
        checkpoint(cancelled);
        const int end = std::min(begin + 128, result.prompt_tokens);
        hidden = model.forward_embeddings(slice_axis(embeddings, 1, begin, end),
            slice_axis(positions, 1, begin, end), {}, cancelled);
        if (event) event("pe_prefill", end, result.prompt_tokens);
    }
    result.prefill_seconds = elapsed(prefill_start);
    return sample_continuation(weights, tokenizer, model, hidden, result, sampling, seed, event,
                               cancelled, layout.next_position);
}

TextGenerationResult generate_edit_images(const Weights &weights, const Tokenizer &tokenizer,
    const std::string &system_prompt, const std::string &user_prompt,
    const std::vector<Tensor> &images, const SamplingProfile &sampling, uint64_t seed,
    const Event &event, std::atomic<bool> &cancelled, LanguageConfig language,
    bool fp32_visual) {
    checkpoint(cancelled);
    require(!images.empty() && images.size() <= 10, "PE-I2I requires 1...10 images");
    require(sampling.max_new_tokens > 0 && sampling.max_new_tokens <= 24000, "invalid PE-I2I token budget");
    token_distribution({0.f}, {}, sampling);
    const int max_prompt = 32768 - sampling.max_new_tokens;
    auto tokens = tokenizer.raw_bounded(prompt_rewrite_chat(system_prompt, user_prompt, images.size()), max_prompt);
    std::vector<VisionInput> processed;
    std::vector<ImageGrid> grids;
    int64_t expanded = tokens.valid;
    for (const auto &image : images) {
        checkpoint(cancelled);
        processed.push_back(pe_vision_input(image));
        const auto &input = processed.back();
        grids.push_back({input.grid_height, input.grid_width});
        expanded += int64_t(input.grid_height / 2) * (input.grid_width / 2) - 1;
        require(expanded <= max_prompt, "PE expanded prompt exceeds token limit");
    }
    (void)image_layout(tokens, grids, max_prompt);
    checkpoint(cancelled);
    VisionConfig config;
    config.pe_qwen35 = true;
    config.pe_fp32 = fp32_visual;
    config.deepstack_layers.clear();
    VisionEncoder vision(weights, config);
    const auto patch_dtype = fp32_visual ? mx::float32 : weights.at("model.visual.patch_embed.proj.weight").dtype();
    std::vector<Tensor> features;
    features.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        checkpoint(cancelled);
        auto patches = mx::astype(processed[i].patches, patch_dtype);
        auto encoded = vision.encode(patches, processed[i].grid_height, processed[i].grid_width,
                                     event, cancelled);
        mx::eval(encoded.merged);
        require(mx::all(mx::isfinite(encoded.merged)).item<bool>(),
                "nonfinite PE-I2I visual features");
        features.push_back(encoded.merged);
    }
    return generate_edit_features(weights, tokenizer, system_prompt, user_prompt,
                                  grids, features, sampling, seed, event, cancelled,
                                  std::move(language));
}
}
