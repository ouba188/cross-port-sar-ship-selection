# 基线对比实验 · 预注册 v1（BATCH 1）

日期：2026-10-06。目的：回答「比融合 / 单极化 / 源训练 / 浅层对齐 / 少标签微调好在哪」。

## 协议（与主线冻结链条同协议，唯一变量 = 方法）
- LOPO，逐港：source = 其余港类分层子样本（min(query_n,|source|)，seed 1000）；target = 本港 query 行。
- 输入特征：VV→`Xs`(128)、VH→`Z`(64)、拼 `[Xs|Z]`(192)，全部冻结缓存复用（pred_cache / pred_cache2）。
- 64 标签：`ann = rng(qpos, 64)`（seed 0），不进金标评价，只作少标签基线输入。
- 单次抽取（非 40-MC），保证 8 臂同抽同判。预算 = 5%（OTH 用）。

## 臂（8）
| # | 臂 | 做法 | 回答 |
|---|---|---|---|
| 1 | VV-only | ridge(Xs, α=1, balanced) | 双极化有用吗 |
| 2 | VH-only | ridge(Z, α=1, balanced) | 同上 |
| 3 | Fusion(=ERM) | ridge([Xs\|Z], α=1, balanced)，全样本用专家 | 路由比直接融合好在哪（**核心，DPIG-Net 式**） |
| 4 | CORAL | 源二阶对齐到目标协方差([Xs\|Z])→ridge | 比浅层对齐好在哪 |
| 5 | MMD | 线性映射 min RBF-MMD(源,目标)→ridge | 同上 |
| 6 | 64-proto | 64 标签类均值→最近邻([Xs\|Z]) | 比直接用 64 标签好在哪 |
| 7 | 64-ridge | ridge 训在 64 标签([Xs\|Z], α=1) | 同上 |
| 8 | OTH(ours) | 冻结 M1→M4 链，natural 5% | 我们的方法 |

## 判据（写死，不据以调参）
- 主指标：macro BA 与加权 ACC **双列**，逐港 + 23 港均值 + 正港数。
- 核心断言（成本效率）：OTH(5% 专家成本) 的 BA ≥ Fusion(100% 成本) 的 BA 的 95%，且 OTH > CORAL/MMD/64-proto/64-ridge。
- 无显著/负向结果如实报，不改协议。

## 批次边界
- BATCH 1（本脚本，全闭合式）：臂 1–8。
- BATCH 2（训练循环，另跑）：DANN / IRM / MixStyle（冻结特征上架小 MLP 适配器）。
