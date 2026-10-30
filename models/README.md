# 已训练研究模型

- `model_bundle.joblib`：预处理器、WOE与逻辑回归、基线/优化LightGBM、校准器和冻结策略阈值。
- 本模型按当前主表实算，不能用于生产授信。
- 只加载本项目可信的joblib文件；该格式的反序列化可执行代码，不要加载未知来源文件。
- 原始记录和客户级预测均不公开打包。
- 训练环境与版本见 `outputs/run_summary.json`、`requirements.txt`。
