import pandas as pd
from sklearn.model_selection import train_test_split

# 讀取資料
input_path = "data/industrial_dataset.csv"
df = pd.read_csv(input_path)

# 去掉第一列（假設該列是多餘的索引）
df_cleaned = df.iloc[:, 1:]

# 提取指定的欄位 (Feature1 到 Feature50 和 Target)
columns_to_keep = [f"Feature{i}" for i in range(1, 51)] + ["Target"]
df_selected = df_cleaned[columns_to_keep]

# 按比例抽樣，例如抽取 10% 的資料
sample_ratio = 0.1  # 設定比例
df_sampled = df_selected.sample(frac=sample_ratio, random_state=42)

# 檢查抽樣結果
print("抽樣後的資料：")
print(df_sampled)

# store
output_path = "data/sampled_data.csv"
df_sampled.to_csv(output_path, index=False)