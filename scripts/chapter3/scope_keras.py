import tensorflow as tf

class MyNetwork(tf.keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # 重み（Variable）をインスタンス変数として保持する
        # ここで一度だけ定義されるため、名前の衝突は起きない
        self.layer1 = tf.keras.layers.Dense(100, activation=None)
        self.layer2 = tf.keras.layers.Dense(50, activation=None)
        self.layer3 = tf.keras.layers.Dense(10, activation=None)

    def call(self, inputs):
        # 同じレイヤー（重みを持つオブジェクト）を順番に通すだけ
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.layer3(x)

# インスタンス化
model = MyNetwork()

# 1回目の呼び出し
i_1 = tf.random.uniform([1000, 784])
output_1 = model(i_1)
print(f"output_1: {output_1.shape}")

# 2回目の呼び出し（同じ 'model' インスタンスを使えば、重みは自動的に共有される）
i_2 = tf.random.uniform([1000, 784])
output_2 = model(i_2)
print(f"output_2: {output_2.shape}")