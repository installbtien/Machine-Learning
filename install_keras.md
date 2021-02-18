# Install Tensorflow and Keras

## NVIDIA Driver
download the driver: https://www.nvidia.com/zh-tw/geforce/drivers/

check your driver version: cmd: C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi

CUDA ToolKit version: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html/ (Table 2)

Tensorflow version: https://www.tensorflow.org/install/source_windows#gpu

## Create an Environment in Anaconda Prompt
```conda
conda create –n tensorflow python=3.6
conda activate tensorflow
```

## Install tensorflow-gpu
```conda
conda install tensorflow-gpu=1.12.0
```

## Test Tensorflow with GPU in Python
```python
import tensorflow as tf
tf.__version__
tf.test.is_gpu_available()
```

## Reference
https://lufor129.medium.com/%E5%82%BB%E7%93%9C%E5%BC%8Ftensorflow-keras%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-730b235275d
