# 模型池预算分配 · 预注册 v1（前置诊断已通过：oracle 天花板 +14.03pp）

日期：2026-10-05。前置：模型池互补性诊断（MODELPOOL_DIAGNOSIS_v1）证明三预测器错误不相交。

## 臂定义（M4 二选一 → 三模型池，其余冻结）
- 池 = {ridge_expert, GBM, MLP}（均源行 [Xs|Z] 训练，类均衡）；默认 = ridge base（Xs）。
- **源侧风险序分配**（部署可用、不用目标标签）：
  1. 源条件表 `cond_m[y, base_pred, m_pred]`（港×类均衡权），对每个候选模型 m 与每行 x：
     期望收益 `u_m(x) = Σ_c 1[c∈observed]·[P(m 对|y=c,base) − P(base 对|y=c,base)]`（observed 来自 64 标签）；
  2. `m*(x) = argmax_m u_m(x)`，效用 `u*(x) = max_m u_m(x)`（>0 才可换）；
  3. 预算 = 5% 名额，按 (priority, −ranking, −u*) 排序取前 k（与 M4 同规则），只有 u*>0 且 m*≠base 的行可入选；
  4. 选中行用 m* 预测，其余用 base。
- 对照臂：oracle 池（用目标标签挑模型，非部署，天花板）、冻结 M4（二选一）。

## 判据（vs 冻结 +1.371pp/22 港/−0.669pp）
1. cp−flat 港级均值 ≥ +1.371pp；
2. 正港 ≥ 22；3. 最差港 ≥ −0.669pp；4. natural ≥ +1.376pp。
全过 ⇒ 模型池成为候选（再外部复验）；部分过 ⇒ 如实描述（含逼近 oracle 的比例）；
全负 ⇒ 负结果入档（注：负则说明"互补性存在但源侧不可分配"是边界结论）。

## 边界
- 不改冻结模块；G 池只有源侧统计与 64 标签，无目标标签泄漏（oracle 臂仅作上界、明确标注非部署）。
- 产物 `sensitivity/novel_arms/pool/`。
