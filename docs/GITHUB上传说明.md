# 这次怎样更新已有GitHub仓库

## 先给老师的文件

发 `reports/final/项目结项总结_5页.pdf`，它正好5页。其他分阶段PDF、结果工作簿和代码放在同一个仓库供核查。

## 不要再上传外面一层文件夹

解压下载的ZIP，进入能直接看见README.md、src、tests、outputs、reports、models的那一层。GitHub最终根目录应直接显示：

```text
credit-risk-analysis/
├── README.md
├── src/
├── tests/
├── data/
├── outputs/
├── reports/
├── models/
├── configs/
├── docs/
├── .github/
├── .gitignore
├── .flake8
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

网页里点Add file → Upload files。从Finder拖入上述内部文件/目录，不是整个外套目录，也不是ZIP。显示隐藏文件用Command+Shift+.，以便上传.gitignore、.flake8和.github。

如果一次文件数太多，分两次：第一次上传src、tests、configs、docs、reports、models、.github以及根目录文件，提交；第二次上传outputs和data，提交。不要因为上传限制把代码改成ZIP。提供的文件包不含原始大数据。

本次建议提交标题：

`feat: 完成阶段二至四离线研究及结项交付`

提交说明：

`补齐训练内特征工程、LR/LightGBM基线、四轮优化、SHAP解释、风险分层与情景价值评估；修订第一阶段变量组合；报告统一PDF并提供5页结项总结。当前为公开申请主表研究版本，历史明细与证券业务指标缺项已登记。`

上传后确认主页能显示README，src下能直接浏览.py，reports/final下能找到5页总结。打开reports后检查PDF能正常预览。

## 清理旧版，避免老师误看

确认新文件和报告齐全后，再删除根目录旧credit_risk_phase.zip、旧Word/Excel及旧credit_risk_phase1目录。不要先删旧版再猜新版是否完整。保留Git提交历史即可，不必把所有旧副本堆在根目录。

更新同一个仓库后地址不变；现有项目链接仍可使用。不要将缺项写成“所有证券数据验证均已完成”。
