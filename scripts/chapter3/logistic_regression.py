import tensorflow as tf
import time

# Apple Silicon等のGPUが認識されているか確認
print("利用可能なGPUの数:", len(tf.config.list_physical_devices('GPU')))

class SimpleLogisticRegression(tf.keras.Model):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        # 変数の初期化
        self.W = tf.Variable(tf.zeros([784, 10]), name="W")
        self.b = tf.Variable(tf.zeros([10]), name="b")
        # オプティマイザの初期化
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    @tf.function
    def call(self, x):
        logits = tf.matmul(x, self.W) + self.b
        return tf.nn.softmax(logits)

    def compute_loss(self, y_true, y_pred):
        # 損失関数
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
        # 勾配の記録とパラメータの更新をクラス内で完結させる
        with tf.GradientTape() as tape:
            predictions = self.call(x)
            loss = self.compute_loss(y, predictions)
        
        # 勾配の計算 (self.trainable_variables で W と b を自動取得)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # 変数の更新
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, predictions


# ---------------------------------------------------------
# 実行・オーケストレーション部分 (メインループ)
# ---------------------------------------------------------
if __name__ == "__main__":
    # データ準備と前処理
    (x_train_full, y_train_full), _ = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(tf.reshape(x_train_full, [-1, 784]), tf.float32) / 255.0
    y_train = tf.one_hot(y_train_full, depth=10)
    
    batch_size = 400
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

    model = SimpleLogisticRegression(learning_rate=0.01)
    training_epochs = 10
    
    print("トレーニング開始...")
    
    # ⏱ 計測開始
    start_time = time.perf_counter()
    
    for epoch in range(training_epochs):
        epoch_start_time = time.perf_counter() # エポックごとの時間も測れます
        
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