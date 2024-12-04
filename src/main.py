import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean

# 读取数据
headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(filepath_or_buffer='../data/data.csv', header=None, names=headers)
# print(data)

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
            print(f'训练次数：{epoch + 1}')
            for i in range(len(x)):
                outputs = self.forward(x[i])
                winner = np.argmax(outputs)
                self.weights[:, winner] += lr * (x[i] - self.weights[:, winner])
            self.plot_weights(epoch, x)

    # 预测
    def predict(self, x):
        outputs = self.forward(x)
        return np.argmax(outputs, axis=1)

    # 绘制权值
    def plot_weights(self, epoch, f):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # 特征
        for i in range(len(f)):
            x = f[i][0]
            y = f[i][1]
            z = f[i][2]
            ax.scatter(x, y, z, color='b', alpha=0.3, marker='x')
        # 权值
        for i in range(self.weights.shape[1]):
            x = self.weights[0, i]
            y = self.weights[1, i]
            z = self.weights[2, i]
            ax.scatter(x, y, z, color='r', marker='o')
        ax.set_xlabel('sepal_length')
        ax.set_ylabel('sepal_width')
        ax.set_zlabel('petal_length')
        ax.set_title(f'Epoch {epoch + 1}')
        plt.show()

# 创建竞争神经网络
model = CompetitiveNN(4, 3)

# 训练竞争神经网络
model.plot_weights(0, train_features)
model.train(train_features, lr=0.01, epochs=10)

# 测试
prediction = model.predict(test_features)
accuracy = np.mean(prediction == test_tags) * 100
print(f'准确率：{accuracy:.2f}%')





