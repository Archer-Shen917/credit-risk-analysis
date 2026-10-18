# 零售客户信用风险数据分析｜第一阶段 V2

**公开数据研究项目 / Data Quality + EDA + WOE/IV + Cross-Segment Risk Signals**

使用 Kaggle Home Credit Default Risk 数据集 `application_train.csv`（307,511 行、122 列）开展第一阶段数据质量与探索性分析。本次 V2 根据评审反馈，补充**全部可用数值特征的分箱 WOE 表与 IV 排序**，并针对 `EXT_SOURCE_2`、`EXT_SOURCE_3` 和 `AMT_CREDIT / AMT_INCOME_TOTAL` 进行二/三变量风险交叉与候选高风险预警识别。

> **数据来源声明：** 数据为公开消费信贷样本，非汇丰银行内部数据，也不包含证券融资融券的持仓、担保比例等必需字段。本阶段数字均为观察性训练数据统计，不是模型的测试集表现、坏账率、策略上线效果，也不能作为真实授信审批规则。历史六张附属表尚未输入，因此多表宽表整合尚未完成。

## 快速开始

本仓库不包含原始大型 CSV。请从 [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) 获取数据，把 `application_train.csv`、`HomeCredit_columns_description.csv` 放入 `data/`（已设 `.gitignore`）。

```bash
python -m pip install -r requirements.txt

# 第一阶段基础数据质量与 EDA
python src/phase1_analysis.py \
  --data data/application_train.csv \
  --dictionary data/HomeCredit_columns_description.csv \
  --output outputs

# V2：全部数值变量 WOE、IV、两/三变量风险交叉
python src/phase1_woe_iv.py \
  --data data/application_train.csv \
  --output outputs

# 基础测试、风格与导入检查
python -m pytest -q
python -m black --check src tests
python -m isort --check-only src tests
```

## 仓库结构

| 目录/文件 | 内容 |
|---|---|
| `src/phase1_analysis.py` | 原主表数据质量、字段字典、相关性、单变量EDA与九张图 |
| `src/phase1_woe_iv.py` | V2数值分箱、WOE/IV、二/三维交叉与稳健性标记 |
| `tests/` | 代码单元测试，覆盖缺失分级、分箱、平滑、缺失编码、哨兵值、最小样本约束 |
| `outputs/tables/iv_numeric_ranking.csv` | **108个数值原始/衍生变量**的IV排序与支持情况（具体变量数以输出为准） |
| `outputs/tables/woe_iv_all_numeric_bins.csv` | 所有可用数值变量的逐箱样本数、标签率、WOE、单箱IV贡献 |
| `outputs/tables/cross_ext2_ext3.csv` | 匿名分数 EXT_SOURCE_2 × EXT_SOURCE_3 二维交叉 |
| `outputs/tables/cross_ext2_ext3_credit_income.csv` | 两匿名分数 × 信用额收入比 三维交叉 |
| `outputs/tables/candidate_high_risk_interactions.csv` | 达到预设支持条件的探索性预警组合 |
| `outputs/figures/10_numeric_iv_ranking.png` | IV 前18名图表 |
| `outputs/figures/11_ext2_ext3_heatmap.png` | 两外部分数交叉标签率热力图 |
| `reports/` | 第一阶段 V2 完整报告（PDF、Word）、综合字段字典（Excel） |
| `data/README.md` | 原始CSV获取与归档说明；不提交原始大表 |

## WOE / IV 计算口径

- `TARGET=1`：数据集提供的还款困难标签；`TARGET=0`：其他记录。
- 数值变量按**不使用标签的五分位数**分箱；离散取值不超过5个的数值变量按具体取值分别成箱；缺失统一独立成 `Missing`。分位点重复时自动减少箱数，极端离散分布退化时以众数/非众数分组；`DAYS_EMPLOYED=365243` 视为特殊编码，单独转为缺失后分析。
- WOE = `ln(P(Good|全体Good) / P(Bad|全体Bad))`；实际计算中对各组的 Good/Bad 人数分别加 0.5 平滑，并按各类别的平滑后总量标准化。WOE **越负**通常表示对应组观察到的坏标签风险越高。
- `IV_bin=(good_share-bad_share)*WOE`；变量 `IV=sum(IV_bin)`。IV 排序不等于因果排序，也不意味着模型一定需要该变量。
- 零正例、零负例、缺失高、样本过少的组会产生不稳定估计；表格包含小样本标志。高度相似的住宅变量存在共线性，IV 不可简单相加。
- **未来建模必需**：按训练集/交叉验证折拟合分箱与WOE映射，仅用训练集统计转换验证集、测试集，防止信息泄漏；未来应加入PSI、时间稳定性和公平性校验。

## 交叉与候选预警条件

- `EXT_SOURCE_2`、`EXT_SOURCE_3` 与 `CREDIT_INCOME_RATIO` 分别以有效记录 P25/P75 划分 Low/Middle/High，空值独立为 Missing。
- 计算分组样本量、`TARGET=1` 观察比例、相对全量基准的风险倍数，及二项式比例的95% Wilson置信区间下界。
- 只有同时满足**每组至少500人、风险倍数≥1.40、Wilson下界高于全样本标签率**时，才标记为**探索性高风险候选组合**；该筛查未进行多重检验校正、时间外验证或因果识别，不能被直接用于真实审批。
- 三个匿名/财务代理变量不等同经银行审核的借款人征信、真实债务收入比，也不代表证券风控的融资担保比例。

## 后续任务

关联 `bureau.csv`、`bureau_balance.csv`、`previous_application.csv`、`POS_CASH_balance.csv`、`credit_card_balance.csv`、`installments_payments.csv`；先对历史表按客户/申请聚合，再以可获取时点约束关联。阶段二才进行训练集划分、WOE 拟合和逻辑回归/LightGBM建模。

## 参考

- [Home Credit Default Risk / Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- 用户提供的《零售客户信用风险数据分析与评分模型构建及策略优化》《编码规范》
