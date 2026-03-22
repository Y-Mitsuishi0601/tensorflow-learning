import tensorflow as tf

# Apple SiliconのGPU (tensorflow-metal) が認識されているか確認
print("利用可能なGPUの数:", len(tf.config.list_physical_devices('GPU')))

class MyNetwork(tf.keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # tf.random_uniform -> tf.random.uniform に変更
        self.W_1 = tf.Variable(tf.random.uniform([784, 100], -1, 1), name="W_1")
        self.b_1 = tf.Variable(tf.zeros([100]), name="biases_1")

        self.W_2 = tf.Variable(tf.random.uniform([100, 50], -1, 1), name="W_2")
        self.b_2 = tf.Variable(tf.zeros([50]), name="biases_2")

        self.W_3 = tf.Variable(tf.random.uniform([50, 10], -1, 1), name="W_3")
        self.b_3 = tf.Variable(tf.zeros([10]), name="biases_3")

    def call(self, inputs):
        output_1 = tf.matmul(inputs, self.W_1) + self.b_1
        output_2 = tf.matmul(output_1, self.W_2) + self.b_2
        output_3 = tf.matmul(output_2, self.W_3) + self.b_3
        return output_3

    def print_parameter_names(self):
        print("Printing names of weight parameters")
        print(self.W_1.name, self.W_2.name, self.W_3.name)
        print("Printing names of bias parameters")
        print(self.b_1.name, self.b_2.name, self.b_3.name)

# ネットワークのインスタンス化（ここで変数が1度だけ作成される）
model = MyNetwork()
model.print_parameter_names()

# tf.placeholder の代わりに、直接テンソルを作成・入力します
# 実際の機械学習では、ここにNumPy配列やtf.dataのバッチが入ります
i_1 = tf.random.normal([1000, 784])
output_1 = model(i_1)
print("Output 1 shape:", output_1.shape)

i_2 = tf.random.normal([1000, 784])
output_2 = model(i_2)
print("Output 2 shape:", output_2.shape)