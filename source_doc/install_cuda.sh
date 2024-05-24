wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb

    1  history
#Installation des Pilotes et Toolkits NVIDIA
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
sudo apt-get -y install cuda-tools-12-4
sudo apt-get -y install cuda-runtime-12-4
sudo apt-get -y install cuda-12-4

#Installation pip3
sudo apt install python3-pip

#Installation des Librairies Python
pip3 install numpy numba pillow joblib py7zr




python3 ./Run_data_2_tiff.py
   24  bash ./make_main.sh
   25  sudo apt install cuda-tools-12-4
   26  sudo apt install cuda-runtime-12-4
   27  bash ./make_main.sh
   28  sudo apt install cuda-12-4
   29  bash ./make_main.sh
   30  sudo apt install cuda-tools-12-4
   31  nvidia-smi
   32  sudo apt list  cuda
   33  sudo apt list  cuda -a
   34  sudo apt install cuda-12-4
   35  sudo apt install cuda
   36  nvidia-smi
   37  bash ./make_main.sh
   38  sudo apt install nvcc
   39  ls /usr/local/cuda
   40  ls /usr/local/cuda/bin
   41  ls -la/usr/local/cuda/bin
   42  ls -la /usr/local/cuda/bin
   43  bash ./make_main.sh
   44  find  /usr/local/cuda  -name  cuda_runtime.h
   45  bash ./make_main.sh