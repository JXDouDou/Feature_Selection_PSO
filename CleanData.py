import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv("data/industrial_dataset.csv")

# 檢查原始資料的第一列名稱
print("原始資料：")
print(df.head())

# 去除第一列 (第一列是多餘索引或空值)
df_cleaned = df.iloc[:, 1:]

# 儲存清理後的資料
df_cleaned.to_csv("data/industrial_data.csv", index=False)

# 確認清理後的資料
print("清理後的資料：")
print(df_cleaned.head())