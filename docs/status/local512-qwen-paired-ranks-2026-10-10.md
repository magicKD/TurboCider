# Qwen LoRA 两路 rank 合并：实Metal通过，大首步形状未稳定盈利

2026-10-10，M4 Max 64GB / macOS 26.6.2。本轮仅增加独立研究 probe、
build script 与 opt-in 测试，不改 production 模型/default route，不下载
或改写原 LoRA。接续[首步通道比例筛选](local512-qwen-prefill-shares-2026-10-10.md)。

## 候选与正确性边界

`tools/native/lora_rank_pair_candidate.hpp` 在一次 GPU dispatch 中消费两个
原始、独立的 A 矩阵，发布两个独立 contiguous F32 rank tensor，让既有
B/scale/delta消费者保持不变。分别筛选跨 threadgroup 的 paired dispatch、
同 threadgroup 的两个 matmul，以及额外4MiB packed A 的参照候选。
一次 dispatch 不等于证实一次物理 X 读取或节省对应带宽。

对本地原 adapter 的32层 gate/up A 共64个 BF16 [256,4096] payload逐一
审计 SHA，64个都不同，不能把某个rank当重复源省略。机器记录保留每个
payload的哈希；component使用真实layer0 A/B，但 X 是合成输入，不是图像
模型推理、实际LoRA全层或整体画质资格。

实Metal自测通过216个 numeric cases 与192个 malformed contracts：
FP16/BF16、M1/33/145、rank64/256、strided X、原source physical offset/
不等pitch、tail、多个 bounded tile。两个F32输出分别与完整F32 oracle
核对，非法几何/dtype/rank/范围明确拒绝。精确核的自测阈值不代表提高了
用户的图像近似要求；在线 delta screening仍按5%预算且要求 finite。

## 实际在线 A→B 筛选

原layer0 BF16 A/B，rank256、K4096；每臂3warmup/15cyclic hot，保留所有
样本，host serial spans。表内毫秒是完整两路 A→B/scale/delta，而非只算
rank kernel。paired及same-TG列均为M16/N64，不按shape事后挑最快tile。

| M / 每路delta N | 当前两路A | paired dispatch | 同TG双matmul | packed A |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 7168 | 1.066875 | 1.023083 | 1.106458 | 1.019083 |
| 1024 / 5120 | .923917 | .930666 | .964833 | .873750 |
| 2096 / 7168 | 1.908833 | 1.898458 | 1.918541 | 1.906583 |
| 2096 / 5120 | 1.562084 | 1.576125 | 1.637625 | 1.601500 |
| 3144 / 7168 | 2.712875 | 2.719666 | 2.766291 | 2.715041 |
| 3144 / 5120 | 2.286958 | 2.304750 | 2.305209 | 2.295292 |

全部筛选的 delta relL2 对当前joint为0，但大首步形状多数持平或变慢。
因此不接入模型、不建立 persistent packed bank；没有把M1024的局部
几十微秒优势乘32层/六步后冒称整请求加速。其他tile及全部原样本均在
原 component receipt；qualification=false，没有 quiet/physical overlap
或真实模型 gain 资格。

## 重现、身份与整理

```sh
TURBOCIDER_NATIVE_OUT=build/qwen-lora-rank-pair \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/local512-qwen-generation-phases-v1-private \
  bash tools/native/build_qwen_lora_rank_pair_probe.sh
build/qwen-lora-rank-pair/qwen-lora-rank-pair-probe

TURBOCIDER_TEST_GPU=1 \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/local512-qwen-generation-phases-v1-private \
  .venv/bin/python -m unittest -v tests.native.test_lora_rank_pair_candidate
```

最终1项opt-in native test通过、无skip，实际包含上述216/192cases；
测试build目录自动回收，并新增180s编译timeout。probe使用保留库rpath，
没有 adjacent dylib，因此 wrapper 的 adjacent_library_sha256 为空；它的
binary前后SHA核对通过，不冒称loaded-image证明。原计时binary与完整
observations/component log仍保留；production库没有新build。

该候选的代码保留在 `tools/native` 的research namespace，测试与实模型
screen分开，不增加应用默认开关。所有本轮owned jobs已terminal，没有
新`.o`/module-cache或需要删除的权重/cache；未动原用户ConvRot草稿。

[机器记录](../design/validation/local512-qwen-paired-ranks-20261010.json)
绑定原adapter payload、component/binary、各shape/recipe的中位与资格边界。
