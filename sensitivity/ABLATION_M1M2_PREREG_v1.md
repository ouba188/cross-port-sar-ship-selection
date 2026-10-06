# M1/M2 消融臂 · 预注册 v1（Helmert vs PCA vs 直接 logits；高斯 vs Student-t）

日期：2026-10-05。用户"开"→ 路线 A 点 4 的两个便宜消融。链内口径、其余冻结、只报告不主张。
这些是**消融对照**（论证冻结选择为何好），不是新模块。

## 臂定义（唯一差异）
- **m1_pca**：决策特征 = 拼 [base|expert] logits 的源行 PCA-14（替代 Helmert 14 维）；
- **m1_direct**：决策特征 = 拼 16 维原始 logits（d=16，不做对比变换）；
- **m2_gaussian**：M2 密度 = 多元高斯（同均值 μ̂_c、同协方差 ψ·shape_c），替代 Student-t。

## 判据（描述性，vs 冻结 +1.371pp/22 港/−0.669pp）
- 主读数 = cp−flat 港级均值；报告正港数、最差港、natural。
- 不设"通过"门槛；结论只写"Helmert/Student-t 相对替代方案的增益/代价"，不进新模块主张。

## 边界
冻结模块文件不改（参数化副本）；产物 `sensitivity/novel_arms/ablations/`。
