# Install Tensorflow and Keras

## NVIDIA Driver

* download the driver: https://www.nvidia.com/zh-tw/geforce/drivers/

* check your driver version: cmd: C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi

```
C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 442.92       Driver Version: 442.92       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1050   WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   43C    P8    N/A /  N/A |    120MiB /  4096MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     12112    C+G   Insufficient Permissions                   N/A      |
|    0     15228    C+G   ...al\Binaries\Win64\EpicGamesLauncher.exe N/A      |
+-----------------------------------------------------------------------------+
```

* CUDA ToolKit version: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html/ (Table 2)

```
CUDA 10.2.89
```

* Tensorflow version: https://www.tensorflow.org/install/source_windows#gpu

```
tensorflow_gpu-2.3.0
```

## Create an Environment in Anaconda Prompt
```conda
conda create –n tensorflow python=3.6
conda activate tensorflow
```

## Install tensorflow-gpu
```conda
conda install tensorflow-gpu=2.3.0
```

### Numpy version

https://github.com/tensorflow/models/issues/9200

https://jennaweng0621.pixnet.net/blog/post/403958360-tensorflow%E8%88%87numpy%E6%90%AD%E9%85%8D%E7%89%88%E6%9C%AC

## Test Tensorflow with GPU in Python
```python
import tensorflow as tf
tf.__version__
tf.test.is_gpu_available()
```

## Reference
https://lufor129.medium.com/%E5%82%BB%E7%93%9C%E5%BC%8Ftensorflow-keras%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-730b235275d
