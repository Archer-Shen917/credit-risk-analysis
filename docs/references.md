# 来源与口径

检索/版本说明日期：2026-10-26。用户材料决定任务范围；外部资料仅用于核对技术实现，不替代原大纲。模型数字均来自本项目实际计算。

1. 用户提供《零售客户信用风险数据分析与评分模型构建及策略优化(2).pdf》，第3页阶段二、第4页阶段三、第5页阶段四。原始场景为证券融资融券与小额股票质押；当前主表来自消费信贷，二者不等价。
2. 用户提供《编码规范(1).pdf》，格式、导入、Google Docstring、类型标注、测试和交接要求；导师截图要求结项总结不超过5页、PDF格式、更新GitHub。
3. 用户上传application_train.csv与HomeCredit_columns_description.csv。TARGET原文定义为至少一笔前Y期还款中迟付超过X天的还款困难，X/Y未公开，不认定为一年期违约。
4. Home Credit官方竞赛与数据说明：https://www.kaggle.com/competitions/home-credit-default-risk/data
5. scikit-learn，Common pitfalls and recommended practices：https://scikit-learn.org/stable/common_pitfalls.html
6. scikit-learn，Probability calibration：https://scikit-learn.org/stable/modules/calibration.html
7. LightGBM，LGBMClassifier（实际运行4.6.0；在线文档可能为更新版本）：https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
8. LightGBM，Parameters：https://lightgbm.readthedocs.io/en/latest/Parameters.html
9. SHAP，TreeExplainer：https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
10. SHAP，force/decision plot介绍：https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html
11. imbalanced-learn，Common pitfalls：https://imbalanced-learn.org/stable/common_pitfalls.html

# 文件证据优先级

`outputs/run_summary.json`与各阶段CSV/JSON是统计事实的机器可读来源；PDF从这些结果生成。Excel是结项快照，内含假设和公式，但不重新训练模型。模型输出不能覆盖数据字典中实际没有的金融含义。
