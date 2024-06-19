import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载波士顿房价数据集
data = pd.read_csv("Boston Housing.csv")

# 查看数据概况
print(data.head())

# 检查缺失值
print(data.isnull().sum())

# 处理缺失值（例如，用中位数填充缺失值）
data["rm"].fillna(data["rm"].median(), inplace=True)

# 将数据分为特征和目标变量
X = data.drop("medv", axis=1)
y = data["medv"]

# 使用相关性分析筛选相关特征
from sklearn.feature_selection import mutual_info_regression

# 计算每个特征与目标变量的相关性
mi = mutual_info_regression(X, y)

# 选择相关性高的特征
selected_features = X.columns[mi > 0.5]

# 使用选定的特征构建新的特征矩阵
X_selected = X[selected_features]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集的房屋价格
y_pred = model.predict(X_test)

# 使用均方误差（MSE）评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("均方误差:", mse)

# 使用决定系数（R^2）评估模型性能
r2 = r2_score(y_test, y_pred)
print("决定系数:", r2)
