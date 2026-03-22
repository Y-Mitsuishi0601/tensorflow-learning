import tensorflow as tf
import time
import datetime # ログフォルダの時刻生成用に追加
import os

print("利用可能なGPUの数:", len(tf.config.list_physical_devices('GPU')))

class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self, hidden_1_units=256, hidden_2_units=128, learning_rate=0.2):
        super().__init__()
        
        def init_weights(shape, name):
            return tf.Variable(tf.random.normal(shape, stddev=0.1), name=name)
        def init_biases(shape, name):
            return tf.Variable(tf.zeros(shape), name=name)

        self.W1 = init_weights([784, hidden_1_units], "W1")
        self.b1 = init_biases([hidden_1_units], "b1")
        self.W2 = init_weights([hidden_1_units, hidden_2_units], "W2")
        self.b2 = init_biases([hidden_2_units], "b2")
        self.W3 = init_weights([hidden_2_units, 10], "W3")
        self.b3 = init_biases([10], "b3")

        # Windowsの場合は tf.keras.optimizers.SGD を使用してください
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    @tf.function
    def call(self, x):
        layer_1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        layer_2 = tf.nn.relu(tf.matmul(layer_1, self.W2) + self.b2)
        logits = tf.matmul(layer_2, self.W3) + self.b3
        return tf.nn.softmax(logits)

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        dot_product = y_true * tf.math.log(y_pred)
        xentropy = -tf.reduce_sum(dot_product, axis=1)
        return tf.reduce_mean(xentropy)

    def compute_accuracy(self, y_true, y_pred):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.call(x)
            loss = self.compute_loss(y, predictions)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, predictions

# ---------------------------------------------------------
# メインループ (TensorBoard連携版)
# ---------------------------------------------------------
if __name__ == "__main__":
    (x_train_full, y_train_full), _ = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(tf.reshape(x_train_full, [-1, 784]), tf.float32) / 255.0
    y_train = tf.one_hot(y_train_full, depth=10)
    
    batch_size = 1000
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

    model = MultiLayerPerceptron(learning_rate=0.2)
    training_epochs = 40
    
    # TensorBoard用のログライターを設定
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join("logs", "gradient_tape", current_time, "train")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    print("ディープラーニング・トレーニング開始 (TensorBoard記録中)...")
    start_time = time.perf_counter()
    
    for epoch in range(training_epochs):
        epoch_start_time = time.perf_counter()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_acc_avg = tf.keras.metrics.Mean()
        
        for batch_x, batch_y in train_dataset:
            loss, predictions = model.train_step(batch_x, batch_y)
            epoch_loss_avg.update_state(loss)
            epoch_acc_avg.update_state(model.compute_accuracy(batch_y, predictions))

        # エポックごとにTensorBoardへ数値を書き込む
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('Accuracy', epoch_acc_avg.result(), step=epoch)
            # ついでに重み(W1)の分布の変化も記録（ヒストグラムタブで見られます）
            tf.summary.histogram('Weights_1', model.W1, step=epoch)

        epoch_duration = time.perf_counter() - epoch_start_time
        print(f"Epoch: {epoch+1:04d} Loss: {epoch_loss_avg.result():.4f} Accuracy: {epoch_acc_avg.result():.4f} ({epoch_duration:.2f}秒)")

    total_duration = time.perf_counter() - start_time
    print("-" * 30)
    print("最適化完了！")
    print(f"合計トレーニング時間: {total_duration:.2f} 秒")