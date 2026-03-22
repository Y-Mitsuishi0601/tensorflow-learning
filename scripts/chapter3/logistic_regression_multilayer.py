import tensorflow as tf
import time

print("利用可能なGPUの数:", len(tf.config.list_physical_devices('GPU')))

class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self, hidden_1_units=256, hidden_2_units=128, learning_rate=0.1):
        super().__init__()
        
        # 【重要】隠れ層を持つ場合、重み(W)はゼロではなく「乱数」で初期化して対称性を壊す
        # 標準偏差(stddev)を小さくしておくことで学習が安定
        def init_weights(shape, name):
            return tf.Variable(tf.random.normal(shape, stddev=0.1), name=name)
        
        def init_biases(shape, name):
            return tf.Variable(tf.zeros(shape), name=name)

        # 第1隠れ層 (入力: 784次元 -> 出力: 256次元)
        self.W1 = init_weights([784, hidden_1_units], "W1")
        self.b1 = init_biases([hidden_1_units], "b1")

        # 第2隠れ層 (入力: 256次元 -> 出力: 128次元)
        self.W2 = init_weights([hidden_1_units, hidden_2_units], "W2")
        self.b2 = init_biases([hidden_2_units], "b2")

        # 出力層 (入力: 128次元 -> 出力: 10クラス)
        self.W3 = init_weights([hidden_2_units, 10], "W3")
        self.b3 = init_biases([10], "b3")

        # M1/M2/M5 Macでは legacy オプティマイザを使用
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    @tf.function
    def call(self, x):
        # レイヤー1の計算 (行列積 + バイアス -> ReLU関数)
        layer_1 = tf.matmul(x, self.W1) + self.b1
        layer_1 = tf.nn.relu(layer_1)

        # レイヤー2の計算 (行列積 + バイアス -> ReLU関数)
        layer_2 = tf.matmul(layer_1, self.W2) + self.b2
        layer_2 = tf.nn.relu(layer_2)

        # 出力層の計算 (行列積 + バイアス -> Softmax関数)
        logits = tf.matmul(layer_2, self.W3) + self.b3
        return tf.nn.softmax(logits)

    def compute_loss(self, y_true, y_pred):
        # 損失関数 (交差エントロピー)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        dot_product = y_true * tf.math.log(y_pred)
        xentropy = -tf.reduce_sum(dot_product, axis=1)
        return tf.reduce_mean(xentropy)

    def compute_accuracy(self, y_true, y_pred):
        # 評価関数
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @tf.function
    def train_step(self, x, y):
        # GradientTapeを使った自動微分
        with tf.GradientTape() as tape:
            predictions = self.call(x)
            loss = self.compute_loss(y, predictions)
        
        # ネットワーク全体(W1, b1, W2, b2, W3, b3)の勾配を一気に計算
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, predictions

# ---------------------------------------------------------
# メインループ
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. データ準備と前処理
    (x_train_full, y_train_full), _ = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(tf.reshape(x_train_full, [-1, 784]), tf.float32) / 255.0
    y_train = tf.one_hot(y_train_full, depth=10)
    
    # バッチサイズの設定
    # バッチサイズを大きくしたときには、Linear Scaling Ruleに従って学習率を調整すると精度を維持できる
    batch_size = 1000
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

    # 2. ディープラーニングモデルのインスタンス化
    model = MultiLayerPerceptron(hidden_1_units=256, hidden_2_units=128, learning_rate=0.2)
    training_epochs = 40
    
    print("隠れ層追加: 第1層=256 Node, 第2層=128 Node")
    print("ディープラーニング・トレーニング開始...")
    
    # ⏱ 計測開始
    start_time = time.perf_counter()
    
    for epoch in range(training_epochs):
        epoch_start_time = time.perf_counter()
        
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_acc_avg = tf.keras.metrics.Mean()
        
        for batch_x, batch_y in train_dataset:
            loss, predictions = model.train_step(batch_x, batch_y)
            
            epoch_loss_avg.update_state(loss)
            epoch_acc_avg.update_state(model.compute_accuracy(batch_y, predictions))

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch: {epoch+1:04d} Loss: {epoch_loss_avg.result():.4f} Accuracy: {epoch_acc_avg.result():.4f} ({epoch_duration:.2f}秒)")

    # ⏱ 計測終了
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    print("-" * 30)
    print("最適化完了！")
    print(f"合計トレーニング時間: {total_duration:.2f} 秒")