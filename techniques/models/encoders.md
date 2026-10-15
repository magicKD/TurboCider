# Encoder：一次性 conditioning 与可证明的缓存

[阶段地图](stage-map.md) · [本地实现](../labs/model_stages.py) · [cache identity](../labs/caches.py)

## 先按实际图分类，不按名字分类

Text encoder 可以是双向注意力，也可以是因果 LLM 或带特定 pooling 的图；vision encoder 可能是 patch transformer/conv；音频前端有 framing/mel/时间位置。必须记录 tokenizer/预处理、mask、positions、padding、hidden-state 取层、pooling、归一化和最终 dtype。不能把“encoder”一概当成双向，也不能把 LLM 的 causal prefix cache 套到双向模型。

一次性 encoder 即使单算子很快，cold load、权重映射、tokenization、首次编译也可能更重要。短提示词的 ANE transfer/launch 固定税可能大于算量；反复同条件生成时，正确缓存整个 conditioning 往往比逐次 offload 更直接。

## 保持语义的方案

**Conditioning cache**：缓存最终供下游消费的 embeddings/mask/positions，不缓存一个与输出取层不同的中间 tensor。key 应包含 source 内容身份、tokenizer/预处理版本、完整有效 tokens/参考图内容、mask、positions、hidden-state recipe、adapters/strength、dtype/layout。按内容而不是文件名/提示词字符串。跨 seed 是否能复用由计算图决定，不是默认把 seed 塞入或移出 key。

**Length/shape bucket**：多个输入可 padding 到小范围 bucket，减少 graph 数和 JIT；有效 query 只能看到原本允许的 keys，pooling 仍定位真实末尾/EOT/有效集合。把 pad hidden 当零也不能代替 mask，因为 softmax 分母仍会改变。视觉 token bucket 还涉及二维 position/patch 边界，不能套 text 长度表。

**GPU 算子**：QKV projection 合并、norm/activation epilogue、online attention、消除 transpose/cast 往返。保持实际 mask、epsilon、position basis 和 cast 点。根据 M/K/N 选择 GEMM，而不是先强行展开权重或把所有 padding 都计算完。

## 关键反例：双向 prefix KV

在双向堆叠中，第 0 层前缀 token 的状态已能看到后缀；因此第 1 层前缀 K/V 也依赖后缀。两个提示词共享 token prefix，不足以复用所有层的 prefix KV。局部第一层投影可复用不代表整个 encoder 前缀可复用。

`encoder_buckets`实际计算一层双向 attention，然后比较下一层 prefix keys：只编码 prefix 与完整编码后截取不同。同时验证 masked padding 与原输出一致、漏 mask 则不同。对于真实 causal encoder，转到 [prefill](llm-prefill.md)的因果证明，而不是从这个负例推成所有 prefix cache 都不可能。

## ANE 和量化放在哪里

长 rows、稳定 shape、FFN 占比高时，可在层内尝试 F-channel split。Conv/固定 vision 子图可做完整 public Core ML 子图，先测输入预处理、layout pack、load、prediction、restore；compute policy 不证明 physical ANE。跨 encoder 层串行依赖不能被两个 queue 名字变成并行。

量化要覆盖短/长文本、特殊 token、多语种、参考图、不同 encoder 输出取层；比较下游生成质量而不只 embeddings cosine。若保留敏感 norm/first/last layers 为高精度，需把实际层策略进入 cache identity。降低输入尺寸/裁 token 是不同语义预算，不能记成纯工程加速。

## 验证与迁移

```sh
python3 -B techniques/labs/run.py stage-encoder
python3 -B techniques/labs/run.py cache-identity
python3 -B techniques/labs/run.py norm-rope
```

测试 padding mask、变化 suffix、source/adapter/preprocess 变动、cache reset、batch 中长度不齐与 pooling。测量 tokenization→最终 conditioning 完成，再测完整请求；释放 encoder weights 时仍保活下游借用的 conditioning。重新探索：新 encoder mask/架构、取层、输入 modality、bucket、quant 或 adapter。
