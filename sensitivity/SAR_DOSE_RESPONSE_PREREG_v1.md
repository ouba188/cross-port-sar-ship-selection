# SAR 原生剂量-响应（因果发现）· 预注册 v1

日期 2026-10-06。目标：在自家 23 港 SAR 上做 do-干预剂量-响应，把「互补头寸 H → 选择增益 G」定律落到本数据。

## 干预（do-操作）
- 专家 = ridge([VV-128 | VH-64])，基座 = ridge(VV-128)。
- **干预 = 衰减 VH 通道**：对 α ∈ {0, 0.125, …, 1.0}（9 档），构造 $\mathrm{expert}_\alpha$ = ridge([VV | α·VH])。
- α=0 ⇒ 专家退化为基座（H=0）；α=1 ⇒ 完整专家（H=H_max）。这是对"第二视图强度"的直接 do-干预。

## 测量与判据（写死）
- 每档 α、每港：头寸 $H(\alpha)=\mathrm{BA}(\mathrm{expert}_\alpha)-\mathrm{BA}(\mathrm{base})$；oracle 选择增益 $G^\star(\alpha)=\mathrm{BA}(\text{oracle 双向})-\mathrm{BA}(\mathrm{base})$。
- 聚合 9 档 × 23 港，得 $H$、$G^\star$ 序列，测 **Spearman(H, G⋆)**。
- **判据**：Spearman ≥ 0.8 且 p < 0.01 ⇒ 「头寸→增益」剂量-响应在 SAR 上成立（因果发现成立）。
- 同时报我们 s2 键的增益 $G_{s2}(\alpha)$ 与捕获率 $\kappa(\alpha)$，看它是否随 H 单调（方法收割头寸的直接证据）。

## 局限（诚实）
- 这是**算法内 do-干预**（衰减第二视图特征），不是对真实 SAR 采集参数的干预；定性方向成立、定量斜率是否跨数据集可搬另行检验（已证伪，2.79×）。
