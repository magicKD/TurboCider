# 当前四家族 benchmark 摘要

完整收敛版见[性能报告](PERFORMANCE_REPORT.md)。2026-10-12，M4 Max64GiB，base-only。
最新重点为[ComfyUI对已有TC最优](COMFYUI_MODEL_COMPARISON.md)：Z1.48×、
Qwen2.1 1.56×、LTX quality1.76×／opt-in近似1.81×；跨窗口回用，LTX单warm，
不是新增原生重测或严格paired排名。下表原同窗口口径保留。

| 完整任务 / 已测兼容路径 | GPU/ANE热态 | 相对参考 |
| --- | ---: | --- |
| Qwen2.1 GGUF512²/40步，对固定Unsloth/Metal | 36.870s | **3.40×，省70.6%** |
| Z-Image Turbo512²/8步，对最快已测Unsloth GPU-text | 5.955s | **3.15×，省68.2%** |
| Qwen2.1 BF16源512²/40步，对mflux0.22 | 31.810s | **1.32×，省24.2%** |
| Z-Image Turbo同组，对mflux0.22 | 5.955s | **1.34×，省25.3%** |
| Z-Image Turbo独立ComfyUI v0.39组，对更快默认MPS 8.535s | 5.840s | **1.46×，省31.6%** |

混合为近似W8A8，不等于严格BF16；cold/内存和Qwen纯GPU略输mflux的
结果保留。FLUX4B当前Q4/g64两fresh3.723s仍只是未配对预检；历史
BF16 GPU/ANE另列。LTX quality GPU63.848s优于INT8混合71.333s，
暂无成功同任务外部分母。9B只作历史，不推算4B。
Comfy组8进程/32PNG/8完整memory；另测SDPA/GPU-only 8.840s更慢，
不采用慢分母；重复prompt改seed确保重新采样，详见[Mac实现与完整对照](COMFYUI_MAC.md)。

报告已按“兼容实测、优势置顶、历史/预检分开”收敛；不继续扩大网格。
