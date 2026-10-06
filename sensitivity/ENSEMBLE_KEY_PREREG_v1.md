# 互补键集成 · 预注册 v1（s2 ∪ margin 是否超单键）

日期：2026-10-05。前置：`MARGIN_KEY`（capture_s2 与 capture_margin 不相关 0.295、Sydney Botany 一负一正 ⇒ 互补）。

## 问题
两互补键集成（z-score 求和排分歧行）是否超任一单键？能否逼近"每港选优"的上界？

## 方法（每港，诊断用 oracle）
- cap_s2、cap_margin（同 MARGIN_KEY）；
- cap_ensemble：分歧行上 (z(s2)+z(margin)) 排序取 top-k，capture（**可部署**，无标签）；
- cap_oracle_pick = max(cap_s2, cap_margin)（oracle 上界，非部署，仅参照）；
- cap_oracle = 真增益 top-k（绝对上界）。

## 判据
1. 主：mean cap_ensemble ≥ max(mean cap_s2, mean cap_margin)（集成不劣于单键）；
2. 次：cap_ensemble 逼近 cap_oracle_pick（报比例 = ensemble/pick）；
3. 逐港：ensemble 在 s2 差的港是否被 margin 补上（Sydney Botany 类）。
