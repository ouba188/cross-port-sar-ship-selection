# 互补键集成 · 结果 v1（s2∪margin 提 capture 16%、达 oracle 上界 81%）

日期：2026-10-05。预注册 `ENSEMBLE_KEY_PREREG_v1.md`。23 港，诊断用 oracle，budget 5%。产物 `ensemble_key.json`。

## 结果
| 键 | capture 均值 |
|---|---|
| s2（类证据，冻结） | 0.281 |
| margin（专家置信度） | 0.251 |
| **ensemble（z-score 求和，可部署）** | **0.327** |
| pick（oracle 逐港选优上界） | 0.402 |

- 主判据过：ensemble 超任一单键（+16% vs s2）；
- ensemble/pick = 0.813（达上界 81%，无标签）；
- 11/23 港严格超两单键；Sydney Botany：s2 −0.222 → ensemble +0.444（margin 补齐）。

## 结论
互补键（类证据 + 专家置信度）集成，可部署地提 capture +16%，达 oracle 上界 81%。
⇒ "capture 由互补信号集决定"成立；集成是第二篇硬方法贡献（或主论文扩展模块）。

## 与今日因果链的关系
紧预算下头寸被封顶 → capture 唯一变量 → capture=键判别力 → 键是互补信号集 → 集成提 capture。
