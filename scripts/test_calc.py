import tensorflow as tf

# どのデバイス（CPUかGPUか）で計算されたかをログ出力する設定をオンにする
tf.debugging.set_log_device_placement(True)

print("行列の掛け算を開始します...")

# 簡単なテンソル（行列）を作成
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])

# 掛け算を実行
c = tf.matmul(a, b)

print("\n計算結果:")
print(c)