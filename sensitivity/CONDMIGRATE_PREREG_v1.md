# 条件关系迁移 · 第 4 模块候选 · 预注册 v1（主线信号做池选择）

日期：2026-10-05。前置：`POOL_FAILURE_ATTRIBUTION_v1.md`（键信息量≈0）、`POOL_PRIORFIX_FINDINGS_v1.md`（非边际漂移）。

## 假设
池分配失败 = 用了**朴素经验对表** `P(m对|y,base)` 作键（源侧统计，跨域条件关系错位）。
主线内部（22/23 港成功）用的是 **M2/M3 类成员似然比 `ratio2`**——它经 class_mass 的 EM 步骤
**投影到目标类分布**（含目标适应）。改用这条已证明有效的信号做模型选择，或能恢复部分条件关系。

## 方法（无标签，仅 64 标签 + 源侧似然 + 目标未标注行）
1. 主线链内量：`ratio2[i,c]` = 目标行 i 对类 c 的证据比（shared_cov → class_mass → project_class_evidence）；
2. 模型一致性：`agree_m[i] = Σ_c ratio2[i,c] · 1[pred_m[i]=c]`（各模型预测与类证据的一致度）；
3. `m*(i) = argmax_m agree_m[i]`，效用 `u[i] = agree_m*[i] − agree_base[i]`；
4. 预算 5%，按 (priority, −ranking, −u) 排序选行；选中行用 m*，其余用 base。

## 对照臂（同 8 港诊断集、同 gbm/mlp 拟合）
base / rand5 / rank5（朴素键，参照）/ **rank5_mc（主线信号）** / truegain5（非部署上界）。

## 判据
- **主**：`rank5_mc − rand5 ≥ 0.5pp`；
- **次**：`rank5_mc − rank5 ≥ 0.5pp`（主线信号胜过朴素键 = 信号本身是关键）；
- 报捕获率 `(rank5_mc − rand5)/(truegain5 − rand5)`，目标 ≥ 25%（朴素键仅 5%）。
全不过 ⇒ 64 标签下的条件关系迁移在当前信号族内不可行（第 4 模块此路不通，诚实记负）。

## 边界
- 只用主线既有的 ratio2 路径，不新增任何超参/模块；不碰目标标签；
- 若 gbm/mlp 的预测与类证据的一致度本身无判别力，负结果成立。
