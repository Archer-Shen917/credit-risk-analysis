# 零售客户信用风险分析与策略研究

**结项版 V3 | Archer-Shen917（Archer Shen） | 2026-10-26**

这是一套基于 **Home Credit 公开消费信贷申请主表**的可复现研究项目。衔接第一阶段V2，完成导师反馈修订，并将阶段二的特征/基线、阶段三的优化/解释/策略和阶段四的条件化价值评估分别交付。

> **范围不能混淆：** 当前只有 `application_train.csv` 和字段说明，没有六张历史明细表，也没有证券账户资产、持仓和担保数据。因此这是“主表条件下的四阶段离线研究闭环”，不是原大纲所有多表/证券指标均已完成，更不是汇丰内部数据、已上线模型或真实经营业绩。

## 先看哪些文件

向导师提交 **`reports/final/项目结项总结_5页.pdf`**，正文恰好5页，不需要把长技术说明当作结项总结。另有1页摘要。

| 位置 | 内容 |
|---|---|
| `reports/phase1/` | V3反馈修订、相关性核查及新业务维度风险交叉，2页PDF |
| `reports/phase2/` | 特征工程、基线模型、编码/不平衡实验，5页PDF |
| `reports/phase3/` | 四轮优化、SHAP、风险分层和公平诊断，5页PDF |
| `reports/phase4/` | 条件化价值、敏感性、上线门槛和交接，4页PDF |
| `reports/final/` | 5页结项总结、1页摘要及结果工作簿 |
| `outputs/phase1-4/` | 实测结果CSV/JSON、图表和审计证据 |
| `models/` | 已训练模型包和LightGBM文本模型，可直接作本地研究评分 |
| `src/credit_risk/`、`tests/` | 可运行源码、单元/烟雾测试 |
| `docs/requirements_matrix.csv` | 逐项对照大纲：已实测、方案级完成、受缺失数据限制 |

## 实际结果（不沿用早期模板值）

307,511条申请，122个原始字段，TARGET=1约8.07%。30%回溯测试集为92,254条。

| 模型 | 测试AUC | KS | AP |
|---|---:|---:|---:|
| WOE逻辑回归 | 0.7072 | 0.3047 | 0.1847 |
| LightGBM基线 | 0.7326 | 0.3417 | 0.2144 |
| 优化LightGBM（校准后） | 0.7336 | 0.3470 | 0.2143 |

精确结果、置信区间和各阈值混淆矩阵以 `outputs/phase3/test_model_metrics.csv` 及 `auc_bootstrap_intervals.csv` 为准。优化较树基线仅小幅变化；概率校准没有改善本次测试Brier。不要写成“AUC达到0.78”“已降低坏账10%”或“已提升银行自动审批20个百分点”。

## 导师反馈落实

- 重点交叉改为 `EXT_SOURCE_2 × EMPLOYMENT_YEARS × CREDIT_INCOME_RATIO`，加入拟合区定义、回溯留出复核。
- 两外部评分彼此相关为约 **+0.1117**，而不是因都与TARGET负相关就互相负相关。加回EXT_SOURCE_3的验证AUC有增量；主方案按可解释维度和缺失率约束排除它，不宣称它已被证实无用。
- 所有正式报告导出为PDF。所有显式函数统一提供 `Args`、`Returns` 和类型注解。
- 根目录直接提供 `src/`、`tests/`、`outputs/`、`reports/` 等；没有额外 `credit_risk_phase1/` 外套目录。

## 数据与复现

建议使用Python 3.11及以上；本次实际运行环境为Python 3.13，具体库版本固定在 `requirements.txt`。原始数据不随公开仓库分发。

把自己有权使用的两份文件放到：

```text
data/raw/application_train.csv
data/raw/HomeCredit_columns_description.csv
```

在项目根目录终端执行（Mac）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m credit_risk.pipeline --data data/raw/application_train.csv --root .
python -m credit_risk.reporting --root .
python -m pytest -q
```

`pipeline`重建数值、模型、CSV和图表；`reporting`从当前结果重建所有PDF。附送Excel是本次CSV结果的整理快照，其中经济情景表有可编辑公式；Excel不是运行训练所需依赖。重跑得到新结果后，以CSV和重新生成的PDF为准，不能把旧Excel当成新的输出。

PDF生成会优先使用系统已安装的中文TrueType字体，不分发字体文件。可通过 `CJK_FONT_PATH` 指向自己的字体；没有兼容字体则使用ReportLab的中文CID回退。不同系统需检查字体与分页。

本地研究评分（不要公开上传逐客户预测）：

```bash
python -m credit_risk.predict --input data/raw/application_test.csv \
  --model models/model_bundle.joblib \
  --output outputs/private/research_predictions.csv
```

只加载自己信任的 `joblib`。评分输出不构成授信建议，不应直接触发自动拒贷或额度调整。

## 训练与评价协议

外层7:3随机分层；70%开发区再分拟合150,679、调参32,289、校准16,144、策略16,145条。最终模型复训仅使用拟合+调参182,968条，校准和策略集不参与模型训练。所有预处理和监督式WOE/IV都在对应训练分区学习。SMOTE只用于拟合区的WOE-LR诊断。

8组网格、正类权重对照和业务交叉优化使用调参集；独立校准集拟合Platt映射，独立策略集冻结两套阈值，最后计算测试指标。代码中Stage 1候选交叉也只在拟合区确定，留出复核在模型选择之后执行。

**已知评估限制：** V2阶段曾观察全样本EDA，因此本测试集不是全新“从未看过”的数据；这是固定的回溯评价，不能代替时间外/外部验证。三折CV是选型后的内部诊断，不是嵌套评估。

## 策略与价值边界

两套三档策略的低风险组标签比例分别约3.33%与4.47%，人工复核占比约24.79%与14.84%。对照规则是研究者定义的示例规则，不是任何银行真实原规则。后续人工60%通过为随机批准假设，不是根据真实标签“精准筛掉坏客户”。

经济情景单列假设金额、年化利差、LGD、人工成本和TARGET到一年PD映射。扩大批准规模会提高敞口及绝对损失代理；模拟净值不是已实现利润、经济资本或因果增益。证券保证金/集中度参数、真实定价和额度需要额外数据及机构审批。

性别/年龄等只用于群体审计，直接字段未进入主模型；群体差异和代理风险仍存在。**不宣称已通过公平、法规或生产部署认证。**

## 测试与代码规范

执行结果以 `outputs/qa/checks.json`、`test_results.txt` 为准。当前已执行真实数值流程、单元/烟雾测试、Python语法编译及函数注释/类型标注检查。

Black/isort/flake8/mypy/pylint提供配置和复查命令。交付环境未安装这些工具且无法连通PyPI，**没有将它们标记为已通过**；首次在本地或CI执行后应根据检查结果整改。`flake8`配置放在其原生读取的 `.flake8`，不是假定它会读取 `pyproject.toml` 中的同名节。

```bash
python -m pip install -r requirements-dev.txt
isort src tests
black src tests
flake8 src tests
mypy src/credit_risk
pylint src/credit_risk
```

工作流内的基本测试可直接运行；工具审计工作流为手动触发，结果不应被误当成本次已经运行的检查。

## 上传GitHub

请看 `docs/GITHUB上传说明.md`。ZIP只用于下载传输；解压后上传其中的目录和文件，不是上传ZIP，也不是再套一层同名文件夹。不要上传166MB原始CSV或 `outputs/private/`。

业务/技术要求映射、复核清单、问题登记、来源和评分边界均在 `docs/`。本项目保留公开数据原作者权利，不因生成代码仓库而改变数据许可。
