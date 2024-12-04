import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

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
        distances = np.zeros((epochs, self.weights.shape[1]))
        # 绘制初始权重
        self.plot_weights(-1, x)
        for epoch in range(epochs):
            epoch_distances = np.zeros(self.weights.shape[1])
            print(f'训练次数：{epoch + 1}')
            for i in range(len(x)):
                outputs = self.forward(x[i])
                winner = np.argmax(outputs)
                self.weights[:, winner] += lr * (x[i] - self.weights[:, winner])
                epoch_distances[winner] = euclidean(x[i], self.weights[:, winner])
            distances[epoch] = epoch_distances
            # self.plot_weights(epoch, x)
        # 绘制迭代后权重
        self.plot_weights(epochs - 1, x)
        # 绘制欧几里得距离
        y = np.arange(epochs)
        for i in range(self.weights.shape[1]):
            plt.plot(y, distances[:, i])
        plt.plot(y, distances.mean(axis=1), color='r', linestyle='--')
        plt.xlabel('euclidean_distance')
        plt.ylabel('epoch')
        plt.show()

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

class Dataset:
    def __init__(self, features, tags):
        self.features = features
        self.tags = tags

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.tags[index]