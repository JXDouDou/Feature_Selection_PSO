import pandas as pd
import matplotlib.pyplot as plt

# 假设你的数据集是一个 DataFrame，df
df = pd.read_csv("data/industrial_data.csv")  # 如果数据来自文件，可以用这个加载

# 提取相关系数
corr_with_target = df.corr()["Target"][:-1]  # 计算特征与目标变量的相关系数，排除 Target 自身

# 打印相关系数
print("Features vs Target Correlation:")
print(corr_with_target)

# 可视化相关系数
plt.figure(figsize=(12, 6))
corr_with_target.sort_values(ascending=False).plot(kind="bar", color="steelblue")
plt.title("Correlation of Features with Target")
plt.ylabel("Correlation Coefficient")
plt.xlabel("Features")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


