import numpy as np
import matplotlib.pyplot as plt

from mathmatics import euclidean

# 竞争神经网络
class CompetitiveNN:
    # 初始化
    def __init__(self, input_size, output_size, random_init=False):
        if random_init:
            self.weights = np.random.rand(input_size, output_size)
        else:
            self.weights = np.full((input_size, output_size), 0.5)
        self.labels = np.zeros(output_size)
        self.threshold = np.zeros(output_size)

    # 前向传播
    def forward(self, x):
        return np.dot(x, self.weights)

    # 训练
    def train(self, x, y, lr_initial=0.01, decay_rate=0.01, epochs=100, use_threshold=False, plot=False):
        distances = np.zeros((epochs, self.weights.shape[1]))
        win = np.zeros(self.weights.shape[1])

        if plot:
            # 绘制初始权重
            self.plot_weights(-1, x, y)

        for epoch in range(epochs):
            lr = lr_initial * np.exp(-decay_rate * epoch)
            epoch_distances = np.zeros(self.weights.shape[1])

            for i in range(x.shape[0]):
                outputs = self.forward(x[i])
                if use_threshold:
                    winner = np.argmax(outputs - self.threshold)
                else:
                    winner = np.argmax(outputs)
                self.weights[:, winner] += lr * (x[i] - self.weights[:, winner])
                epoch_distances[winner] = euclidean(x[i], self.weights[:, winner])
                win[winner] += 1

            distances[epoch] = epoch_distances
            if use_threshold:
                self.update_threshold(win)

            print(f'训练次数：{epoch + 1}，学习速率：{lr:.8f}，平均欧几里得距离：{epoch_distances.mean():.8f}')
            # print(f'胜利次数：{win}')
            # print(f'阈值：{self.threshold}')

            if plot:
                if (epoch + 1) % 100 == 0:
                    self.plot_weights(epoch, x, y)

        if plot:
            # 绘制迭代后权重
            self.plot_weights(epochs - 1, x, y)

            # 绘制欧几里得距离
            t = np.arange(epochs)
            for i in range(self.weights.shape[1]):
                plt.plot(t, distances[:, i])
            plt.plot(t, distances.mean(axis=1), color='r', linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('euclidean_distance')
            plt.show()

    # 更新阈值
    def update_threshold(self, win):
        total_wins = np.sum(win)
        if total_wins > 0:
            win_ratio = win / total_wins
            self.threshold = win_ratio

    # 标记标签
    def mark_labels(self, x, y):
        result = np.zeros((self.weights.shape[1], self.weights.shape[1]))

        for i in range(len(x)):
            outputs = self.forward(x[i])
            winner = np.argmax(outputs)
            # print(f'输入：{x[i]}，输出：{outputs}，输出神经元：{winner}，实际标签：{y[i]}')
            result[y[i], winner] += 1

        self.labels = np.argmax(result, axis=1)
        print('竞争统计结果：')
        print(f'{result}')
        print(f'自动标记竞争神经元：{self.labels}')

    # 预测
    def predict(self, x):
        outputs = self.forward(x)
        return np.argmax(outputs, axis=1)

    # 绘制权值
    def plot_weights(self, epoch, f, l):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        # 特征
        for i in range(len(f)):
            x = f[i][0]
            y = f[i][1]
            z = f[i][2]
            if l[i] == 0:
                ax.scatter(x, y, z, color='r', alpha=0.2, marker='x')
            elif l[i] == 1:
                ax.scatter(x, y, z, color='g', alpha=0.2, marker='x')
            else:
                ax.scatter(x, y, z, color='b', alpha=0.2, marker='x')
            # 权值
        for i in range(self.weights.shape[1]):
            x = self.weights[0, i]
            y = self.weights[1, i]
            z = self.weights[2, i]
            ax.scatter(x, y, z, marker='o')
        ax.set_title(f'Epoch {epoch + 1}')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        # 特征
        for i in range(len(f)):
            x = f[i][1]
            y = f[i][2]
            z = f[i][3]
            if l[i] == 0:
                ax.scatter(x, y, z, color='r', alpha=0.2, marker='x')
            elif l[i] == 1:
                ax.scatter(x, y, z, color='g', alpha=0.2, marker='x')
            else:
                ax.scatter(x, y, z, color='b', alpha=0.2, marker='x')
            # 权值
        for i in range(self.weights.shape[1]):
            x = self.weights[2, i]
            y = self.weights[3, i]
            z = self.weights[0, i]
            ax.scatter(x, y, z, marker='o')
        ax.set_title(f'Epoch {epoch + 1}')
        plt.show()
