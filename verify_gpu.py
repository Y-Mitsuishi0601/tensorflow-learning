import tensorflow as tf

def main():
    print(f"TensorFlow Version: {tf.__version__}")
    
    # 利用可能な物理デバイス（GPU）のリストを取得
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print("\nGPUが正常に認識されています!")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] Name: {gpu.name}, Type: {gpu.device_type}")
    else:
        print("\nGPUが認識されていません。CPUモードで動作します。")

if __name__ == "__main__":
    main()