// Development-only public-API probe: stop at the same pre-norm layer as Z-Image.
#include <llama.h>
#include <ggml-backend.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

struct Capture {
    std::vector<float> values;
    bool captured = false;
};

static bool capture(ggml_tensor * tensor, bool ask, void * data) {
    const bool target = std::strcmp(tensor->name, "l_out-34") == 0;
    if (ask) return target;
    if (!target) return true;
    auto & state = *static_cast<Capture *>(data);
    if (tensor->type != GGML_TYPE_F32) return false;
    state.values.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, state.values.data(), 0, ggml_nbytes(tensor));
    state.captured = true;
    // Returning false stops execution before layer 35, final norm and LM head.
    return false;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: llama-qwen-conditioning MODEL TOKENS RUNS OUTPUT.f32");
        int count = std::stoi(argv[2]), runs = std::stoi(argv[3]);
        if (count <= 0 || count > 512 || runs < 1) throw std::runtime_error("invalid geometry");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        Capture state;
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = 8; cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.cb_eval = capture; cp.cb_eval_user_data = &state;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context load failed");
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = 100 + (i * 7919) % 100000;
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        std::cout << "{\"method\":\"35 layers, pre-norm l_out-34, public callback early stop and CPU readback, fresh KV each sample\",\"samples\":[";
        for (int i = 0; i <= runs; ++i) {
            llama_memory_clear(llama_get_memory(ctx), true);
            state.captured = false;
            auto start = std::chrono::steady_clock::now();
            int status = llama_decode(ctx, batch);
            llama_synchronize(ctx);
            auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (!state.captured || state.values.size() != size_t(count) * 2560)
                throw std::runtime_error("conditioning callback did not produce the expected 2560-wide hidden state; decode=" + std::to_string(status));
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            if (i) std::cout << ',';
            std::cout << "{\"warmup\":" << (i == 0 ? "true" : "false") << ",\"seconds\":" << seconds
                      << ",\"process_peak_rss_bytes\":" << usage.ru_maxrss << "}";
        }
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char *>(state.values.data()), state.values.size() * sizeof(float));
        if (!output) throw std::runtime_error("cannot save conditioning");
        std::cout << "]}" << std::endl;
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
