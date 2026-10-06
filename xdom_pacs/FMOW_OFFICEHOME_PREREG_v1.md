# FMoW + Office-Home 跨域 · 预注册 v1（XDOMAIN 第二篇扩展，放宽口径）

日期：2026-10-05。用户选定：与 SAR 最近的 FMoW（遥感）+ 一般性领域 Office-Home；偏难的排除。

## 结构对应（对照 XDOMAIN_PREREG_v1 §0）
| 数据集 | 域 | 第二视图 | 能检验 |
|---|---|---|---|
| FMoW（WILDS，danielz01 镜像） | 5 大区 LOPO（Africa/Americas/Asia/Europe/Oceania） | **时间元数据**（年内日、小时；与 iWildCam 的 图像+元数据 同构） | P1/P2/P3/P5 |
| Office-Home | 4 域 LOPO（Art/Clipart/Product/Real World，65 类） | 无天然视图 ⇒ 源域监督 R18（同 PACS 方案） | P1/P2/P3/P5 |

## 协议（沿用 PACS 管线，逐字一致）
冻结 A=R50-ImageNet(2048) → base=ridge(A)、expert=ridge([A|B])；B = FMoW 时间元数据（低维标准化）/
Office-Home 源域监督 R18-512。分歧键 = **类别对键**（PACS 补丁版，源侧 cond[y,b,e]）；20% 预算；
rank / rand / 源标定阈值三臂；macro + 加权 ACC 双列。

## 判据（放宽口径：描述性为主）
- P3（支配域上 序>阈值）为主读数；P2（支配域上 序≥随机）次读数；
- 报告逐域表；方向一致的正信号按初步结果保留、如实标注，不因单一阈值关闭方向。

## 边界
不改方法不调超参；下载失败记"未获取"；产物 `E:/Hermes/paper_tracking/xdom_fmow/` 与 `xdom_officehome/`。
