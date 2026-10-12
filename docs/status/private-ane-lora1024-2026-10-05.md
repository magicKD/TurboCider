# Qwen1024 LoRA：限定生成入口与 channel head 接续

接续 [full bucket](private-ane-full-bucket-2026-10-05.md)。原四格 base ≥1.2×、
完整 LoRA/质量/内存/并发与 bandwidth calibration 目标保持 active。

## 实现边界

实现已单独提交为 `e2c4be5`；这些带哈希验证库是working-tree快照，
包含独立ConvRot用户改动，不是clean staged tree重新构建。results仅
暂存本专项planned label片段；其它独立代码/实验继续保留。

原请求验证仅接受512² LoRA。新增默认关闭、严格0/1的
`TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC`，限定1024²、六步、resident、
无参考图/编辑/PE的GPU或`runtime`生成；冻结W8A8与QKV组合不放开。
仍仅一个transformer adapter，原GPU pinned filename/strength/SHA限制
保留，显式runtime原有未资格adapter/strength范围保留；不合入checkpoint。
新入口明确拒绝FP16 rank，保留原FP32低秩投影、BF16 base和Viggle六步
schedule。旧512²不提供此开关时行为不变。

shared predicate用于validation、planned算法label和actual selection；
实际marker在runtime描述替换后追加。合法flag在无adapter的base请求中
不生效，便于同进程回到base；不凭flag宣称实际LoRA执行或质量通过。
screen的`--qwen-lora-1024`在每条GPU/runtime路线上注入同一个opt-in，
校验实际1024尺寸、六步、未合并LoRA、计划/执行标记和FP32选择。非法
模型/尺寸/编辑/frozen/FP16在输出目录创建前拒绝。

## 初版构建与验证

- Private：`build/private-ane-lora1024`，
  `401a447b4e55be67b42c936cf66538007d55ac576381186ceacfba7561ae1182`。
- Public：`build/public-ane-lora1024`，
  `ae91aab4f9e6fd7f51cca1a57906aa26d7005054c02aa45a38c934a972367271`。
- 两库各32项原生Qwen C API contract通过，包括新门、旧512/编辑/其它
  diagnostic边界；1项host helper和57项screen host通过。
- 13 Private host/hardware/MLX、13 Public Core ML/MLX/receipt通过。
  新Public实际flags/class strings/direct links发行guard通过。
- v2模板独立导出到c4224，保留已有c2112和v1。没有修改模型、adapters、
  references或原设计稿；独立ConvRot用户改动仍保留在工作树。

## 实际1024² LoRA初筛

现有Viggle v0.2.1 six-step r256、fox/seed42、strength1，一冷两热，
Private→GPU，两路线独立memory sampling及连续load observation。
相同库/精度/请求，含VAE/PNG的native request wall，排除冷请求；不
借用base或其它库的GPU分母，不使用base-only Q/K融合或FP16 rank近似。

Fa5120/c4224/v2，W8A8 GPU I/O、fixed chunks1、fixed async1、scale cache1、
launch fence1、stage specialize1、LoRA range1，W-prefetch/A8-lookahead0。
227个adapter projections每请求实际绑定；每请求192次Private调用，
0 failed fallback/overflow retry，累计576次narrow/0 full callbacks。
summary complete，load与memory verifier均通过。

GPU hot median=34.090670s，Private=32.516841s。这是约1.048×的正收益
初筛，不是LoRA正式画质/多提示词资格，也不把base的1.258×套过来。
原始目录：`outputs/private-ane-qwen1024-lora-c4224-first/`。
GPU/Private peak footprint=33,343,420,288 / 32,464,925,176 bytes；swap
in/out均为0，仍有compression/decompression，不是完整driver/wired资格。
RGB RMSE=4.016087、PSNR=36.054741dB、SSIM=0.991215、correlation=
0.998523、max abs=87；alpha RMSE=0.177315/max abs=3。未完成latent、
LPIPS/CLIP与语义/多提示词质量验收。

## GPU channel LoRA编译候选

原完整GPU FFN与ANE gate/up修正已经使用request-local编译图，但GPU
channel callback仍逐算子构建。下一隔离库将同一`project_slice`、SiLU、
checkpoint-only down算术编译到request-local图，保持FP32 rank/BF16舍入。
仅channel+LoRA使用；base、Public row、完整GPU fallback和一次full-hidden
down-LoRA不改变。captured share在调用时核验，不复用另一adapter请求。
readiness span还包含attention/input，不是纯LoRA kernel时间。

候选第二库：Private `f8399098537ef86260d23c7ba9b5ff88954510bfa7e58e719309e6339f947f99`，
Public `4d10b2fa00d9485ed854fb7849d53a66c5ea9a922d89ee4ce5601c1a5aaaa7c3`，
目录为`build/{private,public}-ane-lora1024-compiled-head`。各32项Qwen
C API、13 Private及13 Public实机/host回归与新Public发行guard通过。

GPU→Private同库、一冷两热、相同memory/load observer与FP32 rank：
GPU=34.056204s、Private=32.292030s，1.05463×（保守1.05133×）。
目录 `outputs/private-ane-qwen1024-lora-c4224-compiled/`。相对初版Private
热中位仅改善约0.7%，是小幅收益而非新的1.2×LoRA资格；两个热样本
对初版Private的RGB和alpha逐位一致，max uint8 error=0。

旧512²兼容复测（不设1024flag）：GPU=8.051656s、Private=8.178746s，
0.98446×，仍不推荐该Private组合替代GPU。首个热Private输出对既有
512²Private结果RGBA逐位一致，完整图与binding/call/load回执通过；
目录 `outputs/private-ane-qwen512-lora-compiled-compat/`。

Z1024 distill patch现有LoRA、Fa4608/c4224/v2/8步、GPU→Private同库
两热：GPU=36.943939s、Private=32.661560s，1.13111×；两路线memory/load
verifier通过。目录 `outputs/private-ane-z1024-lora-c4224-compat/`。实模型
生成成功不等于正式latent/多提示词/媒体质量验收，冷请求重试另行保留。

Qwen同进程base→原adapter strength1→同adapter strength0.5→base的
四个实际请求均正常退出，累计calls=192/384/576/768，graph load_seconds
相同。首末base RGBA逐位一致，strength1与0.5确实改变输出。原校验
脚本在读取base省略的`lora_applied_projections`时KeyError，因此其summary
保持incomplete；独立CPU复核使用原始`lora_strategy=none`、plan的
`lora_count=0`、calls/图/PNG/byte identity，保存到`verification.json`。
原CLI/请求/PNG/load不重跑或重写。该复核不是性能报告或第二个训练
adapter的质量资格；load原始采样间隙/无竞争已核对，但原start/end
receipt未归档，不将之补成正式独占证据。目录
`outputs/private-ane-qwen1024-lora-shared-state/`。

测量窗口仅对重新核验的既有下载脚本具体进程使用已获授权的SIGSTOP，
未改下载文件或其它外部进程。恢复ledger为
`outputs/private-ane-lora1024-download-pause-ledger.json`；所有inference handles
终止后已重验birth、executable、kernel argv与stopped state，SIGCONT后
观察到不再stopped，新ledger已resumed。原目标仍需最新库512/1024 base
及LoRA、多提示词/反序/更广泛实际
状态切换、质量/内存/device trace和bandwidth-aware calibration。
