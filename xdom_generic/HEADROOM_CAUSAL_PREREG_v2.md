# 头寸→选择增益 因果重做 · 预注册 v2（纠正变量定义）

日期：2026-10-05。纠正 v1 的两处失误：① "头寸"错用均值 H=mean(expert−base)；② α-混合同时改均值与分散度（混杂）。
证据：PACS art 域 H=−0.041（负）却 G=+0.056（正）⇒ 均值不是驱动。

## 正确变量（逐行、诊断用 oracle、非部署）
- **分歧行** D = {i: base_i ≠ expert_i}；
- **可恢复信号（dispersion）**：win_rate = P(expert 对 & base 错 | i∈D)；
- **oracle 头寸**：按真实逐行增益 g_i=1[expert 对]−1[base 错] 选 top-k 分歧行，能拿的最大增益（oracle − base，加权 ACC）；
- **定位效率（capture）**：(rank − rand) / (oracle − rand)。

## 待检验的命题
- **H1**：G(rank−rand) 与 win_rate（分散度）的相关 > 与 mean H 的相关 ⇒ 分散度是正确因果变量；
- **H2**：capture 跨域是否近似常数 ⇒ 律的形态是"键捕获 oracle 头寸的固定比例"；
- **H3**：双因子 `G ≈ capture × oracle头寸` 是否跨数据集（PACS/FMoW）拟合良好。

## 数据
PACS 4 域（K=7）+ FMoW 5 区（K=8、CLIP 视图）。每域报：mean H、win_rate、oracle/rank/rand 增益、capture。
