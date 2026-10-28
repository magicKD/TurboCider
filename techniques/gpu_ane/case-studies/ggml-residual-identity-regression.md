# GGML：residual 身份错误的原始 GPU 回归与负对照

<!-- knowledge
id: technique-gpu-ane-case-studies-ggml-residual-identity-regression
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [few-row MMA](ggml-fewrow-mma-and-reuse.md) · [源码/构建](../sources.md#ggml-fewrow)

2026-10-16，Apple M4 Max 实际执行未修改上游 `test-backend-ops.cpp` 的 `MUL_MAT_ADD(mode=mm2+mm)`。修复版及关闭融合版各通过 28/28；独立副本换回修复前 encoder 文件后 27/28 失败。这说明限定回归能检出原始缺陷，不是只验证自写公式，也不是模型或性能验收。

## 错误是 producer 身份，不是量化比特数

```text
X @ W1.T → mm1 ─┐
                ADD → out
X @ W2.T → mm2 ─┘
```

把其中一个 matmul 与 ADD 融合时，另一个才是 residual。旧 encoder 按“ADD 哪个输入不是 MUL_MAT”猜它；两个输入都是 MUL_MAT 时便无法区分 producer，可能把当前被消去、没有最终写入的临时输出当 residual。

修复按同一 tensor 对象身份找另一个输入。原 fusion checker 本已这样判断，错误在 encoder 绑定阶段，两者合同不一致。[官方 #30100](https://github.com/ggml-org/llama.cpp/pull/30100)及原始 parent/fix 文件 Git blob 均已核对；没有用“低比特误差”解释绑定错误。

GPU/ANE join 也应注意：相同 shape、op enum、名字不等于同一个 producer/slot。残差完成状态、generation、覆盖许可要由对象和 owner 确定；本轮没有执行异构 join。

## 融合检查比节点相邻严格

[`ggml-metal-fusion.cpp` 640–710](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-metal/ggml-metal-fusion.cpp#L640)要求恰好一个 ADD operand 为当前 mm、另一个为 F32 非 weight。最终融合还要求相同 shape、连续布局、精确 MMA gate 和合法内存范围。

ADD 不得覆盖权重/激活；可以恰好 in-place 覆盖 residual，但不能仅部分重叠。通用 graph matching 另外检查 elidable uses/view containment，full 阶段再核 memory ranges。structural reorder 的潜在模式与真正 encode 的可融合模式不同，broad match 不足以直接 encode。

MMA store 在 threadgroup reduction 后加 residual，省一个独立 ADD dispatch；节约多少必须测，不从“存在融合”推出请求加速。

## 原始测试，但不是完整原 root target

固定 ref `988190680d5a89fce97de3c20df2c2813731fd61`。旧参考不动，新 sparse checkout tracked-clean。只在 ignored outputs 复制需要的 GGML subtree、MIT LICENSE 与 test。

原创 8 行 CMake target调用原始 `add_subdirectory(ggml)`，将未修改的 `tests/test-backend-ops.cpp` 链接到原始 GGML。上游 root target 还链接 llama/common，并注明并不需要；本轮不编译模型层，不称原 root build、全部 CI 或发布二进制通过。

两份副本各 179 个 tracked 文件逐字节绑定 Git blob。正向全部对应 fixed ref；负向仅 `ggml-metal-ops.cpp` 来自 pre-fix parent `b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea`，其余仍为 fixed，包括新增模式的原始 test。这是原始旧 encoder / 新原始 test 的组合，不是整个旧版本。

encoder 仅 residual 选择行及注释不同：fixed blob `597d5300c0712eaa589a390d5fba0837812802fd`；旧 blob `2a14cd0c2ddd02945e5f05d6b191ed16c7441e7c`。test blob `89554d4d2bb93792c87d1bce2bdae52864e91d98`。四个 configure/compile receipt 均 exit 0、确认终止；两个 executable、CMake cache、wrapper、复制 manifest 分别绑定。

构建身份陷阱：复制目录无 `.git`，原始 CMake 向上寻找 Git 时读到 TurboCider，生成 `GGML_COMMIT="451161f"`，不是上游 ref。[清单](../sources.md#ggml-fewrow)保存此差异，以逐文件 Git blobs 为身份依据。没有伪造 banner；也不宣称所有系统库已冻结或完全 hermetic。

## 三次运行：成功、对照与失败分账

每进程经 guarded_run 五次一秒总 GPU 占用均为 0% 后启动，最多 180 秒、`-j 1`。开关仅属于 child，不改全局。实际测试后端是 MTL0；另一 CPU device 显示 Skipping，但 CPU 仍被内部创建为比较后端，不能把 `2/2 backends passed` 算两次 GPU 实验。

| 副本 / 开关 | cases | 通过 / 失败 | 实际融合 | 原始 child exit |
| --- | --- | --- | --- | --- |
| fixed、默认融合 | 28 | 28 / 0 | 27 | 0 |
| fixed、进程级 fusion disable | 28 | 28 / 0 | 0 | 0 |
| 旧 encoder、新原始 test、融合启用 | 28 | 1 / 27 | 27 | 1 |

负向 failed/exit 1、guard 的 child_failed 保留。“负对照检出成功”不是“原程序数值通过”。

四类型 Q4_0/Q8_0/Q5_K/F16；N=2/3/5/8/9/13/16，M=1000、K=1024。F16 N=2 低于 MMA 阈值，三个版本都通过。失败 27 项原日志 NMSE 为 `0.951193927–1.052330447`，高于 test 的 `5e-4`。成功行不打印 NMSE，不给它们编造 0 error 或 exact。

日志记录 Apple9/Metal4 capability、SIMD-group matrix=true、has_tensor=false，以及 MMA pipeline、27/0/27 fusion debug/counter。可分发数据只保存去地址化的 kernel base names/统计；含指针的完整 debug logs 留在 ignored outputs。

## 上游 checker 实际保证什么

原 test 建立两个 matmul 与 ADD 的 whole graph，复制同图/输入到 CPU 后端，启用可用 CPU reference mode；每进程内部比较同一随机输入。不同进程从 random_device 初始化，未固定 seed，不是跨版本逐值同输入反事实，更不是 perf A/B。

[`ggml_backend_compare_graph_backend` 2393–2441](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-backend.cpp#L2393)的 whole-graph 模式只检查指定 out。通用框架虽插 sentinel，本模式不据此宣称 guards/OOB 已检查，也不证明全部中间值。NaN/Inf 与 NMSE 按上游 callback 判断，不额外宣称保存了全部 finite 值。

重核日志、28 case 顺序/状态、失败 NMSE、pipeline、fusion、child exit、binary/admission hash。没有 actual/reference 数组，不能独立离线重算 NMSE；与上轮[完整 FC 数组](mldrift-complete-int4-fc.md)的证据能力不同。

## 离线复查与显式复测

已有 artifacts 核验不启动设备：

source capturer 显式禁用 Git lazy fetch；如果 partial clone 缺 blob，会拒绝而非隐式联网。先前获取源码的联网步骤与这个离线核验入口分开。

在自己的新私有 source/build 目录放原 wrapper/源码，使用 `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_CPU_KLEIDIAI=OFF`，再 `cmake --build … --target upstream-backend-ops --parallel 4`。关闭的是 GGML 子项目独立 tests 目录，并非显式加入的原 test。构建不 fetch dependencies、不安装系统工具。

设备复测保留 guard、新目录与超时：

```sh
python3 techniques/gpu_ane/guarded_run.py \
  --output-dir outputs/ggml-mm-mm-replay-admission --timeout 180 -- \
  /usr/bin/env -u GGML_METAL_FUSION_DISABLE -u GGML_METAL_TENSOR_ENABLE \
  GGML_METAL_FUSION_DEBUG=2 \
  outputs/gpu-ane-ggml-fewrow-round13-20261016/build/bin/upstream-backend-ops \
  test -b MTL0 -o MUL_MAT_ADD -p 'mode=mm2\+mm' -j 1
```

不绕过 counter 缺失/忙时拒绝，不并跑变体。错误 abbreviated-SHA 扩展下载失败、lazy Git blob 沙箱网络失败和显式重试记入原始 acquisition note，失败候选不进入来源身份。

未运行真实 Clef 模型、完整测试、perf、M5/MPP、ANE、异构并发、取消/drain 或多请求 slot。粗占用不证明物理独占，编译与 test elapsed 不是 inference latency。
