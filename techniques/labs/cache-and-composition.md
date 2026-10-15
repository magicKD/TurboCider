# 缓存与组合实验：能开关，更要知道改了哪里

[实验室](README.md) · [cache代码](caches.py) · [effect checker](policies.py)

## 1. Exact cache identity

```sh
python3 -B techniques/labs/run.py cache-identity
```

canonicalkey包含模型generation、prompt、有序reference内容SHA、adapter、有效precision/route与sigma。相同路径换bytes、reference重排、换adapter/recipe都使key变化；这是内容cache的最小witness，不是未经证明即可跨seed复用任意prefixKV。

迁移到项目时，用已有verified source/condition identity避免每步重新hash，也不要跳过首次证明；compiledclosure还需防止捕获旧weights。smallmetadata cache与fullW-code bank是不同容量政策。

## 2. DBCache-style残差重用

```sh
python3 -B techniques/labs/run.py residual-reuse
```

toy denoising循环每步计算front；相邻front relative-L1小于阈值才用上次完整middle residual，warmup/last/maxconsecutive保护；跳步或条件变更reset。threshold0路径完整计算，统计actualmiddlecalls与最终trajectory误差。

结果是具体toy的速度代理（省calls），没有GPUtiming或图像质量。可扩展TeaCache累计分数/多项式、TaylorSeer导数、PAB模块payload、分stage阈值；每项先保持off==identity与失效，再按真实step/scene校准。

## 3. Token merge / prune

```sh
python3 -B techniques/labs/run.py token-merge
```

保留conditioningprefix，用mass-weighted合并video token，保存原始group/position映射再unmerge。加权全局sum保持不代表attention等价，unmerge误差单列。可以探索norm/density/semantic/3D tile分组、stale/residual compensation；RoPE和mask沿真实position，不能按新紧凑编号偷换。

## 4. Sana式capability / phase / seam

```sh
python3 -B techniques/labs/run.py technique-composition
```

示例Technique声明`requires/phase/reads/writes/steps`：缺capability、相交schedule的两个exclusiveprecision/backend/token owners、同phase不明确write-read都被指出；不相交step可使用不同precision，off schedule不修改结果。

这类结构检查可以指导工程师/AI组合候选，**不是另一个产品runtime或数值证明**。同一个seam若确实需要多writer，应显式定义顺序和组合语义，再增加新control，不永久禁止组合。

新模型adapter提供attention、FFN、blockloop、denoisesteps、tokenlayout等接缝，技术模块保持通用。来源、改造边界与KernelWiki卡片组织见 [参考设计](../references/kernelwiki-sana.md)，迁移SOP见 [guides/sop](../guides/sop.md)。
