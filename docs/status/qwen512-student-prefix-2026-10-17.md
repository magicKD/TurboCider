# Qwen 2.1：六步 r256 学生的 GPU 精度与前缀复用

2026-10-17，M4 Max / 64 GiB。目标仍保持 active。本轮转向已有的 Viggle v0.2.1
六步学生，没有把 teacher40 的时间、误差或倍率借作它的分母。

## 来源和实际改动

使用原 BF16 Qwen 2.1 encoder / DiT / VAE，加已有的未合并 r256/strength1 adapter：
`Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors`，
SHA256 `2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803`。
文件 1,359,147,904 bytes，实际 454 个 BF16 张量；没有写回/合并模型或下载新权重。
parent 再验 adapter hash，native 仍在绑定前核对发布 pin。

对比两条已有 GPU 算术：original，以及显式 `TURBOCIDER_QWEN21_LORA_BF16_AB=1`
的 joint BF16 A/B operands / F32 ranks / fused-B epilogue。joint 并不是新低比特
权重文件，也不保证同 original 逐位相同。

本轮新增：显式 snapshot=1 可对固定六步学生复用不可变 reference conditioning K/V，
包括 joint 算术以及原有 ref512 编码请求。限制 resident、512²、6 步、发布 pin、
单 adapter、strength1、inference_time、1–3 张标准1024或诊断512参考、GPU-only。
保持原 reference/adapter 门禁；不接 DiT cache、末步 FFN 时间复用、ANE、PE、
streaming、内存预算覆盖、未知 adapter 或独立组合的未测算术开关。

runtime key 已含 adapter identity、实际 BF16-rank/B-epilogue 选择、source、步数和
geometry；新增学生 snapshot recipe identity。snapshot 仍只含完成的静态 K/V 与
条件 owners，不带 target noise、旧目标输出或回调。
[`student FFN reuse` 的既有块纹负结果](local512-student-ffn-reuse-2026-10-08.md)
没有被忽略或重新宣传，本轮未开启该时间近似。

## 39 张真实图片，匹配自己的算术和参考尺寸

所有请求输出 512² RGBA、执行 6 个 sampler 更新、实际 227 adapter bindings，
没有 ANE / encoder ANE / temporal FFN cache / DiT cache。提示词为蓝陶瓷茶壶生成、
将已有蓝茶壶改红并保持壶形和场景；seed73/74。first request、首次新条件与
snapshot populate 单列；计分为同条件热态，不是 fresh prompt 或严格系统冷启动。
没有在测量同时 build、pytest 或比较图片。

第一 cohort：original、joint 各独立 session 11 请求，共 22 PNG。
生成四请求含 warmup、同 seed 前后控制和新 seed；编辑七请求含 warmup、前后
snapshot-off 控制、populate、同/新 seed hit。第二 cohort：标准1024参考下，
6 请求原始→joint→原始同-session 切换，再独立 11 请求 joint，共 17 PNG。

### 六步文生图：配置观察，不是新前缀功能的收益

| 算术 | 三个计分热请求 / s |
| --- | --- |
| original | 8.055 / 8.153 / 8.058 |
| joint | 7.746 / 7.754 / 7.775 |

没有参考前缀，故不能把生成中的约0.3–0.4秒差值算给本轮 snapshot。
它来自已有算术配置及该窗口；精度臂未做 ABBA，不授稳定通用倍率。
对同 seed original 的 joint RGB SSIM：0.987931 / 0.990298，MAE 1.359 / 1.464，
alpha max abs 1 / 2；不是无损，也不是相对 teacher40 的保真证明。

### 六步编辑：不能跨参考编码尺寸借分母

| 参考编码 / 算术 | 匹配 snapshot-off / s | 同 seed hit / s | 新 seed hit / s | 前缀增量画质 |
| --- | ---: | ---: | ---: | --- |
| ref512 / original | 9.610 / 9.574 | 8.306 | 8.274 | SSIM 0.998844 / 0.998669，MAE≤0.379，alpha≤1 |
| ref512 / joint | 9.210 / 9.212 | 7.943 | 7.900 | 两 seed RGBA 逐像素相同 |
| 标准1024 / joint | 13.979 / 13.956 | 8.755 | 8.794 | 两 seed RGBA 逐像素相同 |

同 seed off 分母是前后控制均值，新 seed 为反向匹配控制。
ref512 新前缀减少 original 约13.57%、joint 13.76–14.25%；
标准1024 joint 减少36.99–37.37%。每个 hit 仍仅一项局部观察。
同条路径的前后控制 RGBA 均相同，原有请求 local residual 没有泄漏。

**强基线不能省略：**标准1024下，已有 original 算术＋snapshot hit 在切换窗口
为9.106/9.124秒，joint＋snapshot hit 为8.769秒；相对这个强对照只有约3.8%，
不能把37%说成相对所有已有优化的提升。主要新增能力是让 joint 与 snapshot
同时可用，以及为显式 ref512 学生保留同类复用。

ref512 自身属于独立的参考预处理近似，不把它与标准1024交叉比较。
在同 ref512、无 snapshot 的请求上，joint 对 original 编辑 SSIM 为
0.998529/0.998411，MAE≤0.476、alpha≤1；所以“joint snapshot 增量位级相同”
不等于算术切换也位级相同。图像人工查看符合本批指令，但茶壶合成参考的强纹理、
其他场景/文字/身份/LoRA 质量没有得到通用批准。

## 精度切换、内存和资格

同 session original populate→hit→joint miss→hit→original miss→hit：
每次算术变化确实失效，恢复 original 后 miss 输出和 hit 输出分别与切换前
对应路径 RGBA 完全相同。没有把原始算术的 prefix 借给 joint。

KV bank 的 MLX active 增量：ref512 为555,745,280 bytes（约0.518 GiB），
标准1024为2,166,358,016 bytes（约2.017 GiB）；snapshot off 后均回到对应前值。
仍有原 min(8 GiB, memory ceiling/8) cap / export headroom，非 OS 硬预算。

四个窗口的 process-tree memory sampling 均 complete、system swap-out=0。
ref512 original/joint 峰值31,672,507,120 /31,892,691,816 bytes；
标准1024 switch/joint 为22,918,035,640 /32,180,739,208 bytes。
窗口任务和加载生命周期不同，不能拿这些峰值相减当内存优化；也不包含外部
service/driver 独立归因。四个 continuous CPU-load checks 均失败，保留原件，
`qualification_passed=false`；未停止用户 ComfyUI 进程，默认/App 未改。

## 可复核产物

运行库 `build/qwen512-student-prefix-20261017/libturbocider.dylib`，
SHA256 `b876fc7190cd63259601639ebf1c81c544f6bdc5c17e7f50d1e5bf9db106acff`，
Runtime ID `tc-runtime-build-v1-41879c8b98beff84b17baa7392238cf590e85660fe9a378c72483081e2feddd9`。

- `outputs/qwen512-student-prefix-ref512-20261017/`：22请求、PNG、source snapshot、
  原/joint 分别的控制、内存和负载；`precision-comparison.json` 是独立累计算术比较。
- `outputs/qwen512-student-prefix-ref1024-switch-20261017/`：17请求、PNG、source、
  标准参考和同-session 精度切换回执。保存各自 script identity，不回填旧版本。
- 130 项请求合同：129 通过、1 跳过；真实 MLX lifecycle 通过；C++ options/
  pinned-route gates 通过，新增 student runner 的5项 host tests通过。
- r128 在 predicate/pin guard 中有覆盖，但本轮真实图片只有本地 r256；
  不声称已经实测 r128、ANE 学生、任意 adapter 或全部硬件。

```sh
.venv/bin/python -B tools/validation/qwen_student_prefix_screen.py \
  --library build/qwen512-student-prefix-20261017/libturbocider.dylib \
  --output outputs/new-qwen-student512 --reference-size 512 --seed 73

.venv/bin/python -B tools/validation/qwen_student_prefix_screen.py \
  --library build/qwen512-student-prefix-20261017/libturbocider.dylib \
  --output outputs/new-qwen-student1024 --reference-size 1024 \
  --seed 73 --precisions switch,joint
```

本轮 owned 推理已结束，磁盘约剩19 GiB，未删除模型/用户缓存/旧证据。
后续继续扩展场景、精度/来源对照、安静反序窗口与其他速度候选；完整目标未完成。
