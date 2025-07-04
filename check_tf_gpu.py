import tensorflow as tf

print("TensorFlow 版本:", tf.__version__)

# 检查是否能检测到 GPU
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("未检测到GPU，可能是TensorFlow未正确安装或CUDA环境有问题。")
else:
    print(f"检测到 {len(gpus)} 个GPU：")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # 检查是否为V100
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == "GPU":
            print(f"\n详细信息: {device.physical_device_desc}")
            if "V100" in device.physical_device_desc:
                print("✔ 成功检测到 V100 GPU！")
            else:
                print("⚠ 检测到GPU，但不是V100。")
