# 本地数据，不随仓库公开分发

实际用到：application_train.csv（307,511条）及HomeCredit_columns_description.csv。
请将两份文件置于data/raw/。字段说明含Windows-1252字符，代码显式处理该编码。
原始数据来源为Home Credit Default Risk公开竞赛；须遵守相应条款及取得使用权限。

六张历史表未提供，未进行真实多表聚合；application_test.csv没有TARGET，不用于计算本次测试AUC。
本次测试集从application_train.csv按固定种子分层留出30%。

网页上传不会替你应用本地.gitignore。不要把原始CSV拖入GitHub上传区。
