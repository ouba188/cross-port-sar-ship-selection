# 外部评审（GPT）复核 · 2026-10-06

网页 GPT 读了 README/方法节/基线总表/双向选择/SAR 剂量实验+代码后，提出 3 个风险 + 8 个问题。本文档记录裁决与行动项。

## 三个高风险（GPT 判定，均成立）

1. **评估口径必须统一**。fusion/baseline 用 `query\ann`，双向选择/oracle 用全 query。0.4774 vs 0.4707 现不能直接比。→ **行动**：重跑双向 s2 + oracle 在 `query\ann` 口径下，与其余 11 臂同表。
2. **"因果定律"表述偏强**。`run_sar_dose.py` 算的是 H→G_oracle，方法节却把 G 定义成实际选择器增益；且预注册写了要报 G_s2(α) 但没实现。→ **行动**：补 G_s2(α) 剂量-响应，或把方法节 G 定义改为 G_oracle 并明确区分。
3. **提升幅度小**（s2 vs fusion ≈ +0.7–0.8 BA pp）。强论文取决于显著性 + 标签采样稳定性 + 外部 SAR 泛化 + 机制证据，不能只靠均值。→ **行动**：补 7（显著性）、5（多 seed）、外部 SAR 泛化。

## 8 问裁决（回答已转给用户）

| # | 问题 | 裁决 / 缺口 |
|---|---|---|
| 1 | 数据集 | 24 全球港（23 进评估，Melbourne n=64 跳过），8 类 ~38k 片；另有 F_0921(39 港区/385k) + OpenSARShip3.0*，未进主实验 |
| 2 | 公开集 | OpenSARShip 公开 1.0、FUSAR-Ship；PACS/FMoW=机制验证非主实验 |
| 3 | 特征 | VV=ImageNet R50→源行 PCA-128；VH=e206 域 SSL R18→源行 PCA-64。**e206 SSL 是否接触目标 = 未落协议欠账，须核实披露** |
| 4 | LOPO | 1 港 target + 22 港 source；同船跨域是否严格执行**未验证** |
| 5 | 64 标签 | **固定 seed=0 单抽，须补多 seed** |
| 6 | 超参 | 冻结（τ=8 等），敏感性验证峰顶、未据目标 BA 调；须重申预注册冻结 |
| 7 | 显著性 | **协议有写、未实算，须补 Wilcoxon+bootstrap CI** |
| 8 | 期刊 | TGRS 主、JSTARS 备 |

## 主贡献裁定

- **主贡献 = 类证据门控逐样本模型选择**（方法）。
- **解释性贡献 = headroom→gain 机制发现**（因果定律，诊断/边界层）。

## 行动项（按优先级）

1. 补 64 标签多 seed 稳定性（≥5 seeds），报告 BA/ACC 均值±std。
2. 补 paired significance（Wilcoxon signed-rank）+ bootstrap 95% CI，s2 vs fusion。
3. 统一评估口径 query\ann，重跑双向 s2 + oracle，更新 COMPARISON_MASTER。
4. 补 G_s2(α) 剂量-响应（或改方法节 G 定义为 G_oracle）。
5. 核实 e206 SSL 是否接触目标；核实同船是否跨 source/target；两处泄漏写成显式披露。
6. 外部 SAR 泛化：OpenSARShip 公开 1.0 / FUSAR-Ship 上复跑真链键迁移。
