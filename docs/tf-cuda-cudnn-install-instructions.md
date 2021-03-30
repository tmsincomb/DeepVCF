# Ubuntu 20.04 CUDA 11 & cuDNN 8 for TensorFlow 2.4 install
```bash
# Requirements. PLEASE DO THIS. This is lacking in some instructions and will cause middle steps to fail.
sudo apt install nvidia-cuda-toolkit
nvcc --version

# follow [tensorflow docs](https://www.tensorflow.org/install/source#gpu) for cuda/cuDNN versions needed per tf version
# follow [cuda src](https://developer.nvidia.com/cuda-toolkit-archive) to download that cuda version
# 20.04 deb network had these steps
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# download [cuDNN src](https://developer.nvidia.com/cudnn) to click "Download cuDNN" and become a developer member to install
# download [cuDNN verification]()
# follow [cuDNN docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) to install 
# 20.04 from extracted download from cuDNN src
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
# Test cuDNN install for cuDNN 8
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
# should output Test passed!

# You want the environment because cuda is a global install and tf is not. If you upgrade tf you most likely will break it.
conda create --name tf_2.4 python==3.8
# tf 2.1+ now has tensorflow-gpu and tensorflow combined
pip install tensorflow 
```