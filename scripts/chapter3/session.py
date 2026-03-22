import tensorflow as tf

# ==========================================
# 1. データの準備 (input_data の代替)
# ==========================================
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# 元コードの shape=[None, 784] に合わせるため、28x28を784に平坦化し、正規化
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
# 元コードの one_hot=True に合わせる
y_train = tf.keras.utils.to_categorical(y_train, 10)

# ==========================================
# 2. ミニバッチの作成
# ==========================================
# tf.data.Dataset を使うのが現在の標準的なバッチ処理の書き方です
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=10000).batch(32)

# ミニバッチを1つ（32件分）取り出す
minibatch_x, minibatch_y = next(iter(dataset))

# ==========================================
# 3. 変数の定義
# ==========================================
# random_uniform は random.uniform になり、minval/maxvalを明示します
W = tf.Variable(tf.random.uniform([784, 10], minval=-1.0, maxval=1.0), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")

# ==========================================
# 4. 計算の実行
# ==========================================
# placeholderの代わりに、取り出したデータ(minibatch_x)を直接渡すだけで計算されます
output = tf.matmul(minibatch_x, W) + b

# 結果の確認（即座にテンソルの中身が見られます）
print("出力結果の形状 (Batch Size, Classes):", output.shape)
print("最初の1件の計算結果:\n", output[0].numpy())