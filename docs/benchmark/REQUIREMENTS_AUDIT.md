# 原目标与最新约束逐项核验

2026-10-12。当前权威输入：58组sample catalog、保留的best-local-configs、
对应source/实际请求/媒体/内存证据；不以文档标签或测试绿色替代模型实测。
上一轮是实际进展：增加4B两条兼容参考的8完整PNG/2complete memory及
当前可运行Q4配置，没有重跑TC。当前所有owned实验进程均已terminal。

原目标要求本地base模型、GPU/最佳GPU-ANE、BF16/量化、常见框架和完整
耗时。随后用户明确只看FLUX4B/Z/Qwen2.1/LTX、9B暂缓、外部不兼容者
略过、报告收敛；最新重点为Comfy，TC取已有最优而不充分重测。
以下同时保留原目标和这些约束，不把未测模式删去后称整个扩大目标完成。

| 要求 | 权威证据 | 核验结论 |
| --- | --- | --- |
| 当前四家族/base-only，无LoRA | [总体报告](PERFORMANCE_REPORT.md)、catalog的46active/12历史9B、实际requests与source profiles | 证据支持；不混LoRA/PE overlay或把9B代4B |
| 简洁但详细的docs/benchmark | 主报告、[Comfy重点表](COMFYUI_MODEL_COMPARISON.md)、逐家族/格式/参数页、逐请求JSON与本地链接回归 | 已交付；全表可查样本数/中位数/来源 |
| 原BF16与本地量化表示 | Z五表示16配置、Qwen原Q4_K_M和BF16/mflux affine4/8、4B原savedQ4/g64、LTX ConvRot原组件 | 已有实际完整任务证据；不同recipe不作bit-exact身份 |
| 纯GPU与已有最佳GPU/ANE | Z frozen/Private/Public、Qwen frozen/Private GGUF、LTX GPU/INT8/FP16/Sol/text256；[双列](COMFYUI_MODEL_COMPARISON.md#纯gpu与gpuane一起看不把更慢的混合路线当赢家) | Z/Qwen完整测量，LTX更快GPU明确保留；当前4B Q4混合时延仍未测 |
| 本机参数探索与保存最优配置 | Z14-share/join网格、Qwencache/residency/多精度、LTX8完整配置；best-local-configs的可用当前4BQ4配置及图/库路径 | 已保存实际候选与赢家，不宣称穷尽所有shape/global最优；4B当前Q4混合没有匹配结果 |
| Unsloth | 固定Qwen GGUF Metal、Z原BF16 CPU/GPU text完整对照 | 已测；使用更快Z GPU-text分母，不宣称所有Desktop路径 |
| mflux | 当前Qwen/Z0.22完整组、4B原saved-Q4完整参考热态 | 已测；Qwen pureGPU参考胜出未隐藏 |
| mlx-serve | 当前4B原saved-Q4 4-step SSE/完整PNG热态；LTX原组合memory preflight拒绝 | 4B成功参考已补；拒绝的LTX不强适配或算失败时间分母 |
| Comfy在Mac的实现与性能 | v0.39/PyTorch2.12.1 MPS，Z32PNG、Qwen12PNG、LTX两个97帧AV MP4；完整事件/history/媒体/内存链 | 三家族成功；4B saved-MLX whole index不是stock UNET choice，未形成同格式完整生成，不宣称所有loader都不支持 |
| 完整encoder/DiT/VAE/导出耗时 | PNG/MP4 wall；constructor/cold/fresh/repeat分开；LTX完整视频/音频decode和finite PCM | 已核验；LTX Comfy单warm不是中位数，跨窗口复用不是paired |
| SHA只首次验证 | source-generation注册/metadata guard、nativeQwen跨进程验证bytes0回归；模型产物与raw证据不改写 | 已有证据；不每次inference重扫模型 |
| 宽松delta但保留质量/内存权衡 | LTX最快近似MAE25.03/255、Z/Qwen数学漂移、cold/RAM/swap、有限prompt/seed | 已披露；不当无损、hard-RAM或物理ANE trace资格 |
| 最新“已测TC不再充分测试” | Comfy reference-only、4B后续reference-only的native_retested=false；复用原TC样本/库/精度 | 按要求执行；不把原预检升格为新paired实验 |

## 不能据此证明原扩大实验目标全部完成的项目

**当前4B Q4/g64来源绑定GPU/ANE没有时延数据。** Native GPU的两fresh
3.723469s和两个参考热态是真实已测，但不能替代这项混合对照；旧
BF16 mixed1.628s属于不同来源/旧窗口。当前配置明确保存
`gpu_ane_current_q4_result: null`，未用空值/历史速度制造通过。
要补这项，需要新增来源匹配的artifact/admission和新的原生完整模型
实验；不是重新阅读已有JSON能得到的证据。最新用户要求TC取已有最优，
因此本次核验不自行开新的TC混合网格，也不将这项未知改为成功。

Comfy4B缺少当前同格式完整分母与外部失败路线按用户的兼容政策保留
N/A，不强制下载/转换/改造。更多设备、生产/Public发布、hard-RAM、
physical ANE/native integer MAC trace和任意prompt/shape的全局最优不是
本机报告能证明的资格，也不会靠更换benchmark措辞升级。

当前结论：**报告与Comfy已有最佳记录对照已交付；原扩大目标的全部
实测完成仍未获证明，goal保持active。** 这一页不把测试数量、目录存在
或“compatible”标签当成尚未执行的4B混合性能证明。
本次未运行新模型推理；定向回归226项通过，1组fixture跳过，未运行
全量`make test`。这些回归验证已发布样本/源绑定/边界与报告一致性，
不会证明缺失的当前4B混合时延。

[逐请求catalog](results-20261011.json) · [保留配置](best-local-configs-20261011.json) ·
[详细历史审计](COMPLETION_AUDIT.md)。当前collector-only阶段名恢复有独立
原SSE/PNG/memory proof，旧failed summary保留；inconclusive memory仍拒绝。
