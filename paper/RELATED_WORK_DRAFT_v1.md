# 相关研究综述 · 底稿 v1（近年，2021–2026）

日期：2026-10-06。来源：OpenAlex/Crossref/Semantic Scholar 联合检索（6 个主题 × 12 条）。
定位：我们的方法 = 预算受限的序迁移交接 + 双极化互补视图 + 跨港口 SAR 船舶分类 + 少标签。逐类对账如下。

## A. 跨域 SAR 船舶分类（最近，全是特征对齐族——我们不是）
| 论文 | 年 | 方法 | 与我们的差别 |
|---|---|---|---|
| Unsupervised DA for Ship Classification via Progressive Feature Alignment (TGRS) | 2024 | 渐进特征对齐 | 对齐特征；我们做"路由样本给专家" |
| Multi-Scale Alignment DA for Ship Classification (TASE) | 2024 | 多尺度对齐 | 同上 |
| DA Network for Cross-Imaging Satellites SAR Ship Classification (IGARSS) | 2022 | KL 聚类对齐 | 同上 |
| Unsupervised DA for SAR Ship Detection via Multitask Decoupling (JSTARS) | 2025 | 多任务解耦 | 检测；对齐 |
| Unsupervised SAR Fine-Grained Ship Classification via Spherical Metric Refinement (JSTARS) | 2025 | 球面度量 | 对齐 |
| Multi-Level Alignment Network for Cross-Domain Ship Detection (Remote Sensing) | 2022 | 多层对齐 | 检测；对齐 |
| Improving Deep Subdomain Adaptation by Dual-Branch Network (JSTARS) | 2022 | 双分支子域对齐 | 双分支但做对齐，非路由 |

**结论**：跨域 SAR 船舶分类的现有路线**清一色是特征对齐**（MMD/对抗/渐进/球面度量）。**没有人做"预算受限下把样本路由给互补专家"**——这是我们的空位。

## B. 双极化 SAR 船舶分类（VV/VH 融合族）
| 论文 | 年 | 方法 | 差别 |
|---|---|---|---|
| DPIG-Net：Dual-Polarization Information Guided Network for SAR Ship Classification (Remote Sensing, 44 引) | 2022 | VV+VH 特征**融合** | 直接融合双极化；我们把 VH 当**互补专家、按预算路由**，不是融合 |
| CFAR-DP-FW：CFAR-Guided Dual-Polarization Fusion (JSTARS) | 2024 | CFAR+双极化融合 | 检测 |
| Ship Detection in VH/VV Polarization (Remote Sensing) | 2021 | 轻量 CNN | 检测、可视化 |

**结论**：双极化的价值被公认（DPIG-Net 44 引），但现有做法是**融合**；"把 VH 当作独立互补预测器、在预算内路由交接"没人做。

## C. 预算受限路由 / 自适应计算（非遥感，机制最接近）
| 论文 | 年 | 场景 | 差别 |
|---|---|---|---|
| AM-PPI：Active Multiple-Prediction-Powered Inference | 2026 | 医疗监控推断 | 预算内路由到预测器子集（机制最像），但统计推断非分类、非 SAR |
| CONCUR：Continual Constrained/Unconstrained Routing | 2025 | LLM | 预算路由，LLM 场景 |
| Adaptive Computation Modules (AAAI) | 2025 | CV/ASR | 预算自适应算力 |
| RouterRetriever：Routing over MoE Embedding Models (AAAI) | 2025 | 检索 | 路由到域专家，检索场景 |
| Adaptive Computation in the LLM Era（综述, Cambridge） | 2026 | LLM | 统一路由/级联/选择性预测 |

**结论**：预算受限路由在 LLM/检索/医疗有成熟线，但**从未被引入 SAR 船舶分类、也从未与双极化互补视图 + 少标签结合**。

## D. 置信度校准 / 选择性预测（支撑我们的 capture 结论）
| 论文 | 年 | 发现 | 关联 |
|---|---|---|---|
| Reliable Financial NER Under Domain Shift (2026) | 2026 | 置信度信号在域移下**排序本身会变** | 与"capture=键判别力、校准度逐域变"同源 |

## 净增量裁定（对底稿 §2 相关工作的落点）
1. **空位确认**：预算受限的"样本级路由交接"在 SAR 跨港船舶分类里无人做；
2. **差异化核心**：双极化从"融合"（DPIG-Net）改为"互补专家 + 路由"；
3. **可借力的机制参考**：AM-PPI/CONCUR/ACM 的预算路由思想（非 SAR），我们首次迁移到 SAR；
4. **理论增量**：序可搬误差界 + 互补头寸边界 + 互补键集成，在遥感域适应里无对应。
