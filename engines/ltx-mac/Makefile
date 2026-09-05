CC := clang
CXX := clang++
AR := ar

CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE -I.
OBJCFLAGS := $(CFLAGS) -fobjc-arc
CXXFLAGS := -std=c++20 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -I.
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
ANE_FRAMEWORKS := -framework CoreML
LDLIBS := $(FRAMEWORKS) -lm

BUILD := build
MLX_ROOT ?= $(shell if [ -d vendor/mlx/include ] && [ -d vendor/mlx/lib ]; then echo vendor/mlx; fi)
MLX_CPPFLAGS := -isystem $(MLX_ROOT)/include
MLX_LDFLAGS := -L$(MLX_ROOT)/lib -Wl,-rpath,$(MLX_ROOT)/lib -lmlx -ljaccl
LIB_C := ltx.c ltx_conditioning.c ltx_connector.c ltx_transformer_io.c \
	ltx_latent_stats.c ltx_rng.c
LIB_M := ltx_safetensors.m ltx_weights.m ltx_gpu.m ltx_upsampler.m \
	ltx_video_vae.m
LIB_OBJ := $(addprefix $(BUILD)/,$(LIB_C:.c=.o) $(LIB_M:.m=.o))
CLI_OBJ := $(BUILD)/main.o

.PHONY: all test bench tools ane mlx-smoke mlx-media mlx-pipeline \
	mlx-ane-pipeline clean

all: $(BUILD)/ltx $(BUILD)/libltx.a

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/ltx: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/libltx.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

$(BUILD)/test_ltx: $(BUILD)/tests/test_ltx.o $(BUILD)/ltx.o $(BUILD)/ltx_rng.o
	$(CC) -o $@ $^ -lm

$(BUILD)/test_safetensors: $(BUILD)/tests/test_safetensors.o \
		$(BUILD)/ltx_safetensors.o $(BUILD)/ltx_weights.o
	$(CC) -o $@ $^ -framework Foundation

$(BUILD)/test_gpu: $(BUILD)/tests/test_gpu.o $(BUILD)/ltx_gpu.o
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/test_upsampler: $(BUILD)/tests/test_upsampler.o \
		$(BUILD)/ltx_safetensors.o $(BUILD)/ltx_gpu.o \
		$(BUILD)/ltx_upsampler.o
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/test_video_vae: $(BUILD)/tests/test_video_vae.o \
		$(BUILD)/ltx_safetensors.o $(BUILD)/ltx_gpu.o \
		$(BUILD)/ltx_video_vae.o
	$(CC) -o $@ $^ $(LDLIBS)

test: $(BUILD)/test_ltx $(BUILD)/test_safetensors $(BUILD)/test_gpu \
		$(BUILD)/test_upsampler $(BUILD)/test_video_vae
	./$(BUILD)/test_ltx
	./$(BUILD)/test_safetensors
	./$(BUILD)/test_gpu
	./$(BUILD)/test_upsampler
	./$(BUILD)/test_video_vae

bench: $(BUILD)/bench_linear $(BUILD)/bench_mlp $(BUILD)/bench_qkv \
	$(BUILD)/bench_attention $(BUILD)/bench_cross_attention \
	$(BUILD)/bench_block $(BUILD)/bench_adaln $(BUILD)/bench_upsampler \
	$(BUILD)/bench_video_vae

ane: $(BUILD)/bench_ane_mlp $(BUILD)/bench_block_ane

tools: $(BUILD)/run_connector

mlx-smoke: $(BUILD)/bench_mlx_cpp_smoke

mlx-media: $(BUILD)/bench_mlx_video_vae $(BUILD)/bench_mlx_upsampler

mlx-pipeline: $(BUILD)/bench_block_mlx $(BUILD)/bench_mlx_video_vae

mlx-ane-pipeline: $(BUILD)/bench_block_mlx_ane \
	$(BUILD)/bench_mlx_video_vae

$(BUILD)/bench_mlx_cpp_smoke: tools/bench_mlx_cpp_smoke.cpp | $(BUILD)
	@test -n "$(MLX_ROOT)" || \
		(echo "MLX_ROOT must point to the MLX package root (include/ and lib/)"; exit 2)
	$(CXX) $(CXXFLAGS) $(MLX_CPPFLAGS) $< -o $@ $(MLX_LDFLAGS)

$(BUILD)/bench_mlx_video_vae: tools/bench_mlx_video_vae.cpp \
		ltx_mlx_video_vae.cpp ltx_mlx_video_vae.h | $(BUILD)
	@test -n "$(MLX_ROOT)" || \
		(echo "MLX_ROOT must point to the MLX package root (include/ and lib/)"; exit 2)
	$(CXX) $(CXXFLAGS) $(MLX_CPPFLAGS) \
		tools/bench_mlx_video_vae.cpp ltx_mlx_video_vae.cpp \
		-o $@ $(MLX_LDFLAGS)

$(BUILD)/bench_mlx_upsampler: tools/bench_mlx_upsampler.cpp \
		ltx_mlx_upsampler.cpp ltx_mlx_upsampler.h | $(BUILD)
	@test -n "$(MLX_ROOT)" || \
		(echo "MLX_ROOT must point to the MLX package root (include/ and lib/)"; exit 2)
	$(CXX) $(CXXFLAGS) $(MLX_CPPFLAGS) \
		tools/bench_mlx_upsampler.cpp ltx_mlx_upsampler.cpp \
		-o $@ $(MLX_LDFLAGS)

$(BUILD)/tools/bench_block_mlx.o: tools/bench_block.c \
		ltx_mlx_upsampler.h ltx_mlx_video_vae.h | $(BUILD)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DLTX_ENABLE_MLX_MEDIA -c $< -o $@

$(BUILD)/ltx_mlx_upsampler.o: ltx_mlx_upsampler.cpp ltx_mlx_upsampler.h | $(BUILD)
	@test -n "$(MLX_ROOT)" || \
		(echo "MLX_ROOT must point to the MLX package root (include/ and lib/)"; exit 2)
	$(CXX) $(CXXFLAGS) $(MLX_CPPFLAGS) -c $< -o $@

$(BUILD)/ltx_mlx_video_vae.o: ltx_mlx_video_vae.cpp ltx_mlx_video_vae.h | $(BUILD)
	@test -n "$(MLX_ROOT)" || \
		(echo "MLX_ROOT must point to the MLX package root (include/ and lib/)"; exit 2)
	$(CXX) $(CXXFLAGS) $(MLX_CPPFLAGS) -c $< -o $@

$(BUILD)/bench_block_mlx: $(BUILD)/tools/bench_block_mlx.o $(LIB_OBJ) \
		$(BUILD)/ltx_mlx_upsampler.o $(BUILD)/ltx_mlx_video_vae.o
	$(CXX) -o $@ $^ $(LDLIBS) $(MLX_LDFLAGS)

$(BUILD)/tools/bench_block_mlx_ane.o: tools/bench_block.c \
		ltx_ane_mlp.h ltx_ane_v2a.h ltx_ane_kv.h ltx_ane_qkv.h \
		ltx_mlx_upsampler.h \
		ltx_mlx_video_vae.h | $(BUILD)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DLTX_ENABLE_ANE_MLP -DLTX_ENABLE_ANE_V2A \
		-DLTX_ENABLE_ANE_KV -DLTX_ENABLE_ANE_QKV \
		-DLTX_ENABLE_MLX_MEDIA \
		-c $< -o $@

$(BUILD)/bench_block_mlx_ane: $(BUILD)/tools/bench_block_mlx_ane.o \
		$(BUILD)/ltx_ane_mlp.o $(BUILD)/ltx_ane_v2a.o \
		$(BUILD)/ltx_ane_kv.o $(BUILD)/ltx_ane_qkv.o $(LIB_OBJ) \
		$(BUILD)/ltx_mlx_upsampler.o $(BUILD)/ltx_mlx_video_vae.o
	$(CXX) -o $@ $^ $(LDLIBS) $(ANE_FRAMEWORKS) $(MLX_LDFLAGS)

$(BUILD)/run_connector: $(BUILD)/tools/run_connector.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_linear: $(BUILD)/tools/bench_linear.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_mlp: $(BUILD)/tools/bench_mlp.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_ane_mlp: $(BUILD)/tools/bench_ane_mlp.o \
		$(BUILD)/ltx_ane_mlp.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS) $(ANE_FRAMEWORKS)

$(BUILD)/tools/bench_block_ane.o: tools/bench_block.c ltx_ane_mlp.h \
		ltx_ane_v2a.h ltx_ane_kv.h ltx_ane_qkv.h | $(BUILD)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DLTX_ENABLE_ANE_MLP -DLTX_ENABLE_ANE_V2A \
		-DLTX_ENABLE_ANE_KV -DLTX_ENABLE_ANE_QKV \
		-c $< -o $@

$(BUILD)/bench_block_ane: $(BUILD)/tools/bench_block_ane.o \
		$(BUILD)/ltx_ane_mlp.o $(BUILD)/ltx_ane_v2a.o \
		$(BUILD)/ltx_ane_kv.o $(BUILD)/ltx_ane_qkv.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS) $(ANE_FRAMEWORKS)

$(BUILD)/bench_qkv: $(BUILD)/tools/bench_qkv.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_attention: $(BUILD)/tools/bench_attention.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/tools/bench_attention_ane.o: tools/bench_attention.c \
		ltx_ane_qkv.h | $(BUILD)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DLTX_ENABLE_ANE_QKV -c $< -o $@

$(BUILD)/bench_attention_ane: $(BUILD)/tools/bench_attention_ane.o \
		$(BUILD)/ltx_ane_qkv.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS) $(ANE_FRAMEWORKS)

$(BUILD)/bench_cross_attention: $(BUILD)/tools/bench_cross_attention.o \
		$(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_block: $(BUILD)/tools/bench_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_adaln: $(BUILD)/tools/bench_adaln.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_upsampler: $(BUILD)/tools/bench_upsampler.o \
		$(BUILD)/ltx_safetensors.o $(BUILD)/ltx_gpu.o \
		$(BUILD)/ltx_upsampler.o
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_video_vae: $(BUILD)/tools/bench_video_vae.o \
		$(BUILD)/ltx_safetensors.o $(BUILD)/ltx_gpu.o \
		$(BUILD)/ltx_video_vae.o
	$(CC) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: %.c | $(BUILD)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.m | $(BUILD)
	mkdir -p $(dir $@)
	$(CC) $(OBJCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJ:.o=.d) $(CLI_OBJ:.o=.d) \
	$(BUILD)/tests/test_ltx.d $(BUILD)/tests/test_safetensors.d \
	$(BUILD)/tests/test_gpu.d $(BUILD)/tests/test_upsampler.d \
	$(BUILD)/tests/test_video_vae.d \
	$(BUILD)/tools/bench_linear.d \
	$(BUILD)/tools/bench_mlp.d $(BUILD)/tools/bench_ane_mlp.d \
	$(BUILD)/tools/bench_block_ane.d $(BUILD)/ltx_ane_mlp.d \
	$(BUILD)/ltx_ane_v2a.d $(BUILD)/ltx_ane_kv.d \
	$(BUILD)/tools/bench_qkv.d \
	$(BUILD)/tools/bench_attention.d \
	$(BUILD)/tools/bench_attention_ane.d $(BUILD)/ltx_ane_qkv.d \
	$(BUILD)/tools/bench_cross_attention.d \
	$(BUILD)/tools/bench_block.d $(BUILD)/tools/bench_adaln.d \
	$(BUILD)/tools/bench_upsampler.d $(BUILD)/tools/bench_video_vae.d \
	$(BUILD)/tools/run_connector.d $(BUILD)/bench_mlx_cpp_smoke.d \
	$(BUILD)/bench_mlx_video_vae.d $(BUILD)/bench_mlx_upsampler.d \
	$(BUILD)/tools/bench_block_mlx.d $(BUILD)/ltx_mlx_upsampler.d \
	$(BUILD)/tools/bench_block_mlx_ane.d \
	$(BUILD)/ltx_mlx_video_vae.d
