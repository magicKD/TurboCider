# vpipe headroom：缩放必须穿过 bias、LoRA 和非线性

<!-- knowledge
id: technique-gpu-ane-case-studies-vpipe-bias-lora-and-headroom
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [vpipe控制器](vpipe-controller-and-failure.md) · [五文件读本](../sources.md#consumer-handoff) · CPU数学见证

固定 `tgo-app-dev/vpipe@c2989a76841888e4d8d208fad4f476c487710962`，2026-10-16重新追踪stage→emitter→retry→restore及一个具体VAE调用点。参考checkout tracked-clean，只读；没有执行vpipe、触发真实overflow或测模型质量。并行ANE专题已有相关线索，本页以重新核对的源码范围与独立CPU见证推进，不认领其原worker实验。

## 1. 不能只看“缩小up再乘回”这一行

读本 [`ane-ffn.cc:300–580`](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L300) 对projection index1应用 `1/_up_scale`，先在FP32合成base与LoRA，再cast FP16；bias loop同样只缩index1。emitter决定index1的含义：

| 图 | weight slots，零起始 | bias slots | index1缩放对应什么 |
| --- | --- | --- | --- |
| SwiGLU | 0 gate，1 up，2 down | 3 bg，4 bu，5 bd | up与bu缩小；bd不在这一分支里 |
| GELU | 0 up，1 down | 2 bu，3 bd | down与bd一起缩小 |

[`ane-emitter.cc:183–221,368–448`](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc#L183) 明确列出feature顺序，并在down projection之后将bd加到最终模型输出。weight slot从activation之后的feature开始，对应关系另核对了`ane-module.cc:447–512`的实际slot创建/fill。

[`ane-ffn.cc:743–850`](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L743) 非finite重试时把slot1及`nproj+1`的bias再除4，增长sticky scale，随后对**整个返回输出**乘回scale，再写BF16。早期chunk不重算，后续chunk使用新scale。

这五段必须合起来读，不能从scale变量名或注释的“exact”直接继承数值无损资格。

## 2. SwiGLU的最终bias为何是条件性风险

固定row-vector、权重[out,in]约定：

```text
G = X Wgᵀ + bg
U = X Wuᵀ + bu
H = SiLU(G) ⊙ U
Y = H Wdᵀ + bd
```

若只令 `Wu,bu → (Wu,bu)/s`，则H缩小s倍，但模型仍在down之后加原bd。恢复整个输出得到：

`s * (H Wdᵀ/s + bd) = Y + (s−1)bd`。

这是理想实数链的确定性偏移；FP16 staging/计算、BF16写回另有舍入误差，不能声称真实runtime必精确等于这条公式。

条件必须同时成立：实际进入scale>1的状态、图有最终bias且该bias非零、没有别处补偿。scale=1、bd=0或补偿正确时，不能据此宣称同样故障。

CPU见证使用两输出bias `[0.5,-0.25]`，s=4的偏移为 `[1.5,-0.75]`，s=4096时为 `[2047.5,-1023.75]`，而输出仍全finite。按chunk使用s=1/4/16、bd=0.5时，偏移分别0/1.5/7.5；数学row独立不保证row之间的数值误差配方一致。

合理修复候选是在模型内同时令bd→bd/s，或把恢复放在最终bias之前，然后再加原bias。两者都需重新验证精度、graph/cache身份及实际consumer；本轮没有修改参考实现或取得修复后的模型质量结果。

## 3. 具体VAE入口不是完全抽象的组合

[`metal-minimax-h3-video-vae.cc:1060–1082`](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/minimax-h3/metal-minimax-h3-video-vae.cc#L1060) 设置`o.biases=true`，未改Options默认`gelu=false`，因此走SwiGLU。`1110–1123`又将w1两半作为gate/up，w2作为down，并把各Linear的非空bias指针传入source。

所以bias-enabled SwiGLU不是只有接口声明而从无调用的假想配置。不过本轮没有读取该模型checkpoint、测bd非零率，也没有让这个VAE触发headroom重试；只能确认**源码中存在对应调用组合和条件性风险**，不能发布“已复现VAE坏图”或影响比例。

进一步读到`1305–1435`的真正consumer：GPU头部走SwiGLU，ANE尾部原地覆盖norm buffer，join后继续learned per-channel gated residual；未见在这个边界按headroom scale撤销额外bias。因下游gate/residual/norm会继续变换差异，`(s−1)bd`描述FFN输出边界，不是最终像素误差。该调用点也把`finish`返回值转void；这是又一个需故障注入的失败发布入口，而非本轮已触发的运行时故障。

模块的`Options::biases`默认false；不能把这条VAE观察套给MiniMax transformer、所有vpipe模型或所有ANE执行。

## 4. GELU不能照搬SwiGLU的缩放位置

GELU不是正齐次函数，通常 `s*GELU(z/s) ≠ GELU(z)`。正确的实数变换是在GELU之后把down权重和其bias一起除s，再恢复输出。

所读GELU的slot1恰好是down、bias slot3恰好是bd，因此这一路不同于SwiGLU final-bias问题。CPU多维见证对正确down缩放误差为0；故意将scale移到up/非线性之前，即使也正确补偿最终bias，该case仍有约0.37185最大绝对误差。

这说明projection index必须带图类型解释。重构统一FFN接口时若把“第二个矩阵”永远称up，便可能在数值变换位置上出错。

## 5. LoRA必须先合成，再执行同一缩放

源码的正确顺序值得保留：解码base到FP32、加 `Σ α B A`、最后整体乘mul再cast。若只缩base、却把未缩的adapter delta加回，恢复时就将delta额外放大s倍；CPU见证隔离了这个错误顺序，输出差异约0.01648。

LoRA的B行布局不能照搬base row view。读本有独立`b_stride/b_offset`，banded adapter还按 `(br/group)%parts` 选择A的rank分区。1024-row band把一批delta变成BLAS矩阵乘，减少逐row rank-axpy调用；带宽/并发收益和“是否实际用AMX”仍不是本轮CPU数学见证能证明的。

SwiGLU中合成后的gate不应被任意缩放穿过SiLU；合成后的up及其bias应遵守同一scale。最终bias补偿、LoRA合成、quantizer的affine bias不是同一个参数，不要都叫“bias”便共同处理。

## 6. 即使补偿正确，也不是bitwise无损承诺

2的幂缩放在一定浮点范围内可以简单表示，但不能恢复已经underflow掉的位。独立binary16例子：原值`2^-20`非零，除4096得到`2^-32`，收窄binary16为0；再乘4096仍为0。这个例子没有模拟整vpipe图，也没有声称特定ANE flush/subnormal策略。

此外本项目输出BF16路径是截断而非RNE，上一轮已单列。输入本身BF16→FP16溢出也不是仅缩up权重就一定能解决；非finite检测只是拒绝/重试门禁，不能成为完整质量门禁。

因此每个真实checkpoint/adapter至少要对照：GPU原source语义、scale1、触发后sticky scale、bias非零/零、带LoRA、chunk转换位置和最终residual。不能只检查输出finite或一个局部GEMM。

## 7. 离线重放与证据范围

headroom_contracts.py为原创标准库FP64代数参考，只有明确的underflow小例使用binary16 pack；不导入上游或设备库。

通过的CPU数学合同与源码调用链加强了迁移前的反证能力，不等于上游实际故障复现。真实VAE、LoRA、overflow重试和请求质量仍待受控执行。
