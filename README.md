# Cross-Port SAR Ship Classification via Headroom-Guided Model Selection

跨港口 Sentinel-1 SAR 船舶分类：**少标签逐样本模型选择（类证据门控）**。本仓库是论文的实验代码、冻结模块源码、预注册/结论文档与正文活源的快照。

## 现状一句话

方法 = **双向模型选择**：两个源训岭分类器（基座 VV、专家 VV⊕VH）之间，用 64 个目标标签导出的「类证据键」逐样本决定信谁。它在 macro BA 上**超过恒用专家（fusion）和全部 DA 基线**，并发现一条**跨域不变的方向性因果定律**：互补头寸 → 选择增益。

## 方法（M1–M4 冻结链）

- **M1** `fit_source`：源域 14 维 Helmert 决策特征上拟合类条件高斯（共享协方差 + 各类均值）。
- **M2** `shared_covariance_likelihood`：把 64 标签并入，Student-t 预测似然。
- **M3** `class_mass` + `project_class_evidence_rt`：EM 估计目标类先验，熵正则 OT 投影得类证据比 ρ。
- **M4** `robust_scores` + 双向选择：保守证据差键 s2 的**符号**决定基座 vs 专家（分歧样本上）。

## 闭环因果链（每环有 SAR 实测）

```
第二视图(VH) ──▶ 互补头寸 H ──do-干预剂量-响应──▶ 选择增益 G
                                              ▲
                            键判别力(AUROC_s2) ──┘
                                              ▲
                        专家置信度校准(AUROC_conf) ──┘
```

| 环节 | 证据 |
|---|---|
| 双极化提供头寸 | VV 0.361 → VH 0.418 → Fusion 0.471（BA） |
| **H → G（主定律，do-干预）** | SAR Spearman 1.0000 (p<1e-6)；PACS 0.9515；FMoW 同向 |
| 键判别力 → 捕获率 | Spearman 0.807 (p=3.2e-6) |
| 专家校准 → 键判别力 | 0.526 (p=0.010) |
| 选择 ATE（键 vs 随机） | +1.885pp，23/23 港 |
| 方法收割 | 双向 s2 BA 0.4774 > fusion 0.4696；capture 45% of oracle |

**序迁值不迁**（统一框架）：定律的方向迁、斜率不迁（2.79×）；键的序迁、值不迁（命题 1）。方法只依赖序（top-k / 符号），阈值与定量律都断在同一处。

## 对比实验（13 臂，23 港 LOPO）

| 方法 | BA | ACC | 类别 |
|---|---|---|---|
| VV-only | 0.361 | 0.383 | 单极化 |
| VH-only | 0.418 | 0.415 | 单极化 |
| Fusion(=ERM) | 0.471 | 0.493 | 融合 |
| CORAL | 0.405 | 0.456 | 浅层 DA |
| MMD | 0.314 | 0.281 | 浅层 DA |
| DANN | 0.379 | 0.418 | 深度 DA |
| IRM | 0.408 | 0.486 | 深度 DA |
| MixStyle | 0.423 | 0.495 | 深度 DA |
| 64-proto | 0.442 | 0.589 | 少样本 |
| 64-ridge | 0.439 | 0.614 | 少样本 |
| OTH（单向路由，旧） | 0.457 | 0.462 | 路由 |
| **双向 s2（本方法）** | **0.477** | **0.501** | 模型选择 |
| oracle 双向（上限） | 0.523 | 0.539 | 参照 |

- 主指标 macro BA：本方法第一；ACC 上仅 64-ridge 超（多数类倾斜、BA 崩，作 limitation）。
- 深度对齐族（MMD/DANN/IRM/MixStyle）全证负，坐实「特征对齐不是本问题的解」。
- 负结果留档：模型池、少样本修正（M5）、因果因子挖掘（不变性排序）、源侧 g 预测器，均关闭。

## 目录

```
src/                      冻结模块源码（M1–M4 的 bdh_pixel_loss_controls + 证据投影）
sensitivity/              主实验：基线对比、双向选择、键消融、剂量-响应、诊断
xdom_pacs/                PACS 跨域复现（真链键迁移 P2 2/2）
xdom_generic/             FMoW/OfficeHome/CLIP 跨域复现 + 头寸因果
paper/                    正文活源（方法节 + 总表 + 相关工作底稿 + 裁定）
```

## 关键结论档（paper/ 与 sensitivity/ 内）

- `paper/SECTION3_METHOD_v2_ZH.md` — 方法节活源（含 §3.4 因果链 + §3.5 序/值框架）。
- `sensitivity/SAR_DOSE_RESPONSE_FINDINGS_v1.md` — 因果发现闭合（Spearman 1.0）。
- `sensitivity/COMPARISON_MASTER_v1.md` — 13 臂对比总表。
- `sensitivity/BASELINE_COMPARE_FINDINGS_v1.md` — 基线对比结论。

## 待办（定稿前）

详见 `paper/EXTERNAL_REVIEW_20261006.md`（外部评审复核，6 条行动项）。核心：

1. 补 64 标签多 seed 稳定性（≥5 seeds）+ paired significance（Wilcoxon + bootstrap CI），s2 vs fusion。
2. 统一评估口径（query\ann）重跑双向 s2 + oracle，与其余臂同表。
3. 补 G_s2(α) 剂量-响应（或改方法节 G 定义为 G_oracle）。
4. 核实并披露两处泄漏：e206 SSL 是否接触目标、同船是否跨 source/target。
5. 外部 SAR 泛化：OpenSARShip 公开 1.0 / FUSAR-Ship 复跑真链键迁移。
6. 补 EN 方法节；写实验节把因果链逐条落地。

## 环境

Windows / Git Bash；Python = ClearSAR（`D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe`，GDAL+torch+sklearn+scipy+POT）。torch 跑需 `KMP_DUPLICATE_LIB_OK=TRUE`。数据（pred_cache/*.npz、xdomain_data）不随仓库分发。
