# 真链跨域复验（FMoW / Office-Home）· 结果 v1

日期：2026-10-05。预注册 `XDOM_FULLCHAIN_PREREG_v1.md`。K=8 前 8 高频类、主线真链（ratio2 键）、20% 预算。

## FMoW（5 区，K=8，视图 B = 采集元数据）
P1 5/5、P2 5/5、P3 1/5 —— **但全部退化，不能算数**：
| 区 | base | expert | rank | rand | 判读 |
|---|---|---|---|---|---|
| Africa | 0.7063 | 0.7075 | 0.7075 | 0.7075 | rank=rand=expert |
| …5 区同型 | | expert−base 仅 +0.1~+1.25pp | | | 预算未绑定 |

**根因**：K=8 取易类后，采集元数据视图 B 几乎无增益（expert−base ≤1.25pp），分歧数 < 20% 预算 ⇒
三臂退化成"全换 expert"，P2=5/5 平凡成立。⇒ **FMoW 不是"键是否迁移"的有效检验**（视图 B 太弱，无余量可挑）。

## Office-Home（4 域，K=8，视图 B = 源域监督 R18）
（见 `fullchain_officehome_result.json`，跑完后填）

## 结论（诚实）
- PACS 是唯一视图 B 强到有充足余量、能干净检验键迁移的跨域数据；
- FMoW/OH 的视图 B 太弱（元数据 / 弱 R18），即便真链也"无米下锅"，退化为平凡；
- 故"真链键跨域成立"目前**只被 PACS 证实**，FMoW/OH 既不能证实也不能证伪（退化）。
