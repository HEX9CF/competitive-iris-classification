import numpy as np
import pandas as pd

# 读取数据
headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(filepath_or_buffer='../data/data.csv', header=None, names=headers)

print(data)

# 数据预处理
data['class'] = data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 分离特征和标签
features = data.iloc[:, :-1].values
tags = data.iloc[:, -1].values

# 归一化
features = (features - features.mean(axis=0)) / features.std(axis=0)

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train_features, test_features = features[:train_size], features[train_size:]
train_tags, test_tags = tags[:train_size], tags[train_size:]

# 竞争神经网络
class CompetitiveNN:
    # 初始化
    def __init__(self, input_size, output_size):
        self.weights = np.full((input_size, output_size), 0.5)

    # 前向传播
    def forward(self, x):
        return np.dot(x, self.weights)

    # 训练
    def train(self, x, lr=0.01, epochs=100):
        for epoch in range(epochs):
            for i in range(len(x)):
                outputs = self.forward(x[i])
                winner = np.argmax(outputs)
                self.weights[:, winner] += lr * (x[i] - self.weights[:, winner])

    # 预测
    def predict(self, x):
        outputs = self.forward(x)
        return np.argmax(outputs, axis=1)

# 训练竞争神经网络
model = CompetitiveNN(4, 3)
model.train(train_features, lr=0.01, epochs=100)

# 测试
prediction = model.predict(test_features)
accuracy = np.mean(prediction == test_tags) * 100
print(f'准确率：{accuracy:.2f}%')





