## 出力例

```bash
% uv run scope1.py 

利用可能なGPUの数: 1
2026-03-22 10:47:06.191336: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M5
2026-03-22 10:47:06.191353: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 32.00 GB
2026-03-22 10:47:06.191357: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 12.48 GB
2026-03-22 10:47:06.191472: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2026-03-22 10:47:06.191670: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Printing names of weight parameters
W_1:0 W_2:0 W_3:0
Printing names of bias parameters
biases_1:0 biases_2:0 biases_3:0
Output 1 shape: (1000, 10)
Output 2 shape: (1000, 10)

```