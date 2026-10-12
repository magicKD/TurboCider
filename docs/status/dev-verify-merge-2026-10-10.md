# dev 合并至 dev-verify：M4 Pro 轻量验收（2026-10-10）

## 分支关系与整合

同步远端后，合并前 `dev-verify` 为 `8d12b44cc764d079ee31c6092725e22fcc8fb9e0`，
`dev` 为 `ea55048933b6c53be44289b515418ec2a9756dca`。
共同祖先为 `4fd6fff7232e97df4a0abacf9cebae3fd54c5175`。
`git rev-list --left-right --count dev-verify...dev` 为 **43 / 39**：dev 有39个独有提交，
dev-verify也有43个独有提交，不能直接快进或以一方覆盖另一方。
dev相对共同祖先涉及322个文件、约4.7万新增行，其中约3.1万行为文档；
39个提交不等于39项已取得性能资格的产品功能。

在dev-verify上整合19个冲突文件。保留该分支的普通LoRA严格目标绑定、
前缀快照、DiT缓存预设、M5低内存门禁、异步图准备、S1/A8及异常drain，
接入dev的source leases、共享LoRA ranks、转换码缓存、混合GGUF、编码器融合和BF16流式加载。
本次为本地合并，不推送远端，不替换已安装App或生成发布包。

## 新能力及适用边界

- **原始BF16低内存生成**：编码器与DiT分阶段加载、双槽预取、GPU最后读者退役。
  当前显式开启，限512²基础GPU生成，不能与LoRA、编辑、命名DiT缓存或融合编码器混用。
  本机内存/速度取舍见下表；管理的权重预算不是整个请求硬RAM上限。
- **Qwen混合Q4_K_M GGUF**：Q4/Q5/Q6导入、局部embedding、并行解码及GPU/显式Private ANE路径。
  小型实际MLX混合格式投影、所有权和取消测试通过；本机没有完整模型对，未做整模型验收。
- **GPU编码器与LoRA优化**：融合prefill、验证后复用tokenizer、可选权重保留、共享低秩投影、
  B epilogue/BF16 operands和转换权重缓存。小型数值/生命周期测试通过；实验开关保持显式。
  其他M4 Max记录中的加速比例不能套用到本机。
- **诊断与稳定性**：source generation/held-fd校验、分阶段计时、内存/实际调用回执、
  取消与文件变更检测、资源回收。App、Swift/C绑定和服务相对原dev-verify没有源代码改动，
  本次收益主要在执行层。

## 本机短步对照

Apple M4 Pro，48GiB物理内存，macOS26.6 (25G72)。仅使用本机已有的
Qwen-Image-2.1 BF16、Viggle v0.2.1六步r128 LoRA和FLUX.2 Klein4B，没有下载模型。

每项独立进程/持续C ABI会话，512²、GPU、同种子/请求，串行执行。
每项仅两次：首次会话请求/后续请求。基础生成使用两条不同光照提示词，均不命中conditioning缓存；
其他项重复提示词。秒数包括导出，不包括engine constructor。
变化列只比较第二次请求，**每臂只有一个后续样本，不能证明稳定加速或普遍回退**。
更早的40步旧基线只保留raw，不纳入本表；收到轻量化要求后的所有新推理最多8步。

| 场景 | 合并前，首次 / 后续 s | 合并后，首次 / 后续 s | 后续耗时变化 | 全图比较 |
| --- | ---: | ---: | ---: | --- |
| Qwen基础生成，8步 | 23.57 / 23.57 | 23.01 / 23.19 | -1.60% | 两张PNG SHA/完整RGBA一致 |
| Qwen r128 LoRA，6步 | 20.60 / 15.39 | 20.56 / 15.53 | +0.93% | 两张PNG SHA/完整RGBA一致 |
| Qwen单参考编辑，6步 | 24.56 / 19.44 | 24.46 / 19.30 | -0.69% | 两张PNG SHA/完整RGBA一致 |
| FLUX.2 Klein4B，4步 | 6.31 / 4.55 | 6.04 / 4.49 | -1.35% | 两张PNG SHA/完整RGBA一致 |

普通路径在这组短测中基本持平，未观察到明显回退，也没有证实稳定提速。
编辑使用同一张旧版生成图作为参考、512参考编码、component_staged以及同一r128适配器。
旧版resident编辑在48GiB被保守内存准入拒绝；使用已有component_staged后成功，没有放宽门禁。

**编辑质量仍有已有局限**：请求要求将狐狸放到木屋旁，输出仍主要是森林中的狐狸、对比度发生变化，
没有出现所要求的木屋。合并前后像素完全一致，故该样例未显示新回归，但运行成功不等于编辑指令达标。
6/8步输出用于功能/回归检查，不替代完整质量验收，不能称所有功能已经完善。

### 新增原始BF16流式路径

同合并库、同两条8步提示词/seed29，与普通resident控制对比：

| 路径 | 首次 / 后续 s | 进程生命周期峰值 GiB |
| --- | ---: | ---: |
| 普通resident | 23.01 / 23.19 | 28.62 |
| BF16 streaming，prefix28 / weight13GiB | 34.77 / 30.38 | 13.67 |

本窗口峰值降低约**52.3%**，后续请求慢约**31.0%**；两张完整PNG SHA及RGBA均与普通路径一致。
这为笔记本提供可选的低内存模式，不能宣称普遍加速。
峰值取每进程lifetime physical footprint的两次最大值；系统VM计数不能归因该进程，
不声称整机无内存压力或整请求硬RAM上限。

每次完成8个实际denoise步骤；encoder完成36层、8次suffix fills/fences，
DiT完成256次层执行、32次fills/fences，均drained。后续请求两个component的native验证缓存均命中。
首个流式请求包含完整source SHA验证；不同生命周期成本计入整请求，不拼接成kernel加速倍率。

关键配置：

```text
TURBOCIDER_QWEN21_BF16_STREAMING=1
TURBOCIDER_QWEN21_BF16_STREAM_ENCODER_PREFIX=28
TURBOCIDER_QWEN21_BF16_STREAM_DIT_PREFIX=28
TURBOCIDER_QWEN21_BF16_STREAM_WEIGHT_GIB=13
```

请求使用 `execution=gpu`、`residency=component_staged`、`qwen21_dit_cache=off`、512²、8步，
无LoRA或参考图。13GiB仅为管理的权重预算。

## 构建与关键检查

原生库/CLI与App重新编译通过；私有ANE冲突代码额外以启用宏通过语法检查。
系统Xcode动态加载器异常，改用本机已有CommandLineTools/SDK15.2，没有新工具安装。
App重编译及Qwen工作流测试通过，未做桌面手动操作或发布包验收。

| 检查 | 最终结果 |
| --- | --- |
| `make test-qwen21` | 65项中7项skip；Qwen公开契约44项通过；prompt parser、sampling、重编译Swift工作流通过。Qwen35 sample模块整类skip |
| `make test-runtime-ane-host test-quantized-host` | 17+24+36+23项完成；量化/GGUF中13项GPU/可选资源相关skip，随后选择关键项单独开启GPU测试；worker契约也通过 |
| 全部公开请求契约 | 128项，3项缺本地LTX相关资源skip，其余通过 |
| 仓库边界/布局 | 17项通过 |
| 小型MLX/Metal关键检查 | 10项最终通过，缓存2项单独复测；共享LoRA ranks、workspace/外部base/精度切换、S1转换码缓存、GGUF packed bank、BF16流式、编码器64组typed内核、LoRA lease、校准/SmoothQuant bridge |

流式小模型覆盖encoder/DiT全张量严格一致、prefix0/1/2、三个Euler/KV pass、双槽跨层复用、
预算不足、取消、源文件截断与闭包生命周期。新增拒绝“流式+命名DiT缓存/融合编码器”的回归，
避免静默忽略请求。权重码缓存新增S1同generation命中、换generation或去掉S1重新转换的覆盖；
LoRA新增工作区与显式共享ranks/外部base等价及精度recipe切换失效检查。

初次失败日志保留：合并后的计时变量名不一致已修正；新增测试夹具矩阵字段顺序已修正；
旧源代码字符串断言适配函数抽取/Qwen encoder capability；测试启动改回项目venv，
Pillow复用其他本机项目的已有安装。最终通过结果不覆盖这些原始失败。

依用户轻量化要求，未跑整个 `make test`、长程压力/完整画质campaign、所有模型e2e或Private ANE整模型性能验收。
本机没有完整Q4_K_M模型对，Z-Image缺完整text encoder；没有下载补齐。

[可移植机器记录](../design/validation/dev-verify-merge-light-20261010.json)保存完整样本、库SHA、
全图SHA/像素比较及流式回执。原始报告、PNG、失败与构建日志保存在gitignored的
`outputs/dev-verify-merge-20261010/`。
