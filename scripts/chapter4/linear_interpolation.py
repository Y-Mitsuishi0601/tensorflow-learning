import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


n_hidden_1 = 256
n_hidden_2 = 256


def load_mnist():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = x_test.reshape((-1, 784))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return x_test, y_test


def build_model():
    inputs = tf.keras.Input(shape=(784,))
    hidden_1 = tf.keras.layers.Dense(
        n_hidden_1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=(2.0 / 784) ** 0.5
        ),
        bias_initializer="zeros",
        name="hidden_1",
    )(inputs)
    hidden_2 = tf.keras.layers.Dense(
        n_hidden_2,
        activation="relu",
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=(2.0 / n_hidden_1) ** 0.5
        ),
        bias_initializer="zeros",
        name="hidden_2",
    )(hidden_1)
    outputs = tf.keras.layers.Dense(
        10,
        activation="relu",
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=(2.0 / n_hidden_2) ** 0.5
        ),
        bias_initializer="zeros",
        name="output",
    )(hidden_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def load_tf1_checkpoint_weights(model, checkpoint_path):
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_map = reader.get_variable_to_shape_map()

    # helper to find a checkpoint variable name by suffix
    def find_var_by_suffix(suffix):
        for name in var_map:
            if name.endswith(suffix):
                return name
        return None

    name_map = {
        "hidden_1": ("hidden_1/W", "hidden_1/b"),
        "hidden_2": ("hidden_2/W", "hidden_2/b"),
        "output": ("output/W", "output/b"),
    }

    for layer in model.layers:
        if layer.name not in name_map:
            continue
        kernel_suffix, bias_suffix = name_map[layer.name]
        ck_kernel_name = find_var_by_suffix(kernel_suffix)
        ck_bias_name = find_var_by_suffix(bias_suffix)
        if ck_kernel_name is None or ck_bias_name is None:
            available = list(var_map.keys())[:40]
            raise KeyError(
                f"Cannot find variables for layer '{layer.name}' in checkpoint {checkpoint_path}. "
                f"Searched suffixes '{kernel_suffix}', '{bias_suffix}'. Available (sample): {available}"
            )
        kernel = reader.get_tensor(ck_kernel_name)
        bias = reader.get_tensor(ck_bias_name)
        layer.set_weights([kernel, bias])


def interpolate_weights(opt_weights, rand_weights, alpha):
    return [
        (1.0 - alpha) * opt + alpha * rand
        for opt, rand in zip(opt_weights, rand_weights)
    ]


x_test, y_test = load_mnist()

opt_model = build_model()
rand_model = build_model()

checkpoint_path = "frozen_mlp_checkpoint/model-checkpoint-550000"
load_tf1_checkpoint_weights(opt_model, checkpoint_path)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
writer = tf.summary.create_file_writer("linear_interp_logs")

results = []
alphas = np.arange(-2, 2, 0.1, dtype=np.float32)
opt_weights = opt_model.get_weights()
rand_weights = rand_model.get_weights()

for step, a in enumerate(alphas):
    interp_weights = interpolate_weights(opt_weights, rand_weights, a)
    opt_model.set_weights(interp_weights)
    logits = opt_model(x_test, training=False)
    cost = loss_fn(y_test, logits).numpy()
    results.append(cost)
    with writer.as_default():
        tf.summary.scalar("cost", cost, step=step)

plt.plot(alphas, results, "ro")
plt.grid()
plt.ylabel("error")
plt.xlabel("alpha")
plt.show()