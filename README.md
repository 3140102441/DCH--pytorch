# DCH
CV Project ：PyTorch implementation for DCH: [Deep Cauchy Hashing for Hamming Space Retrieval](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf), Yue Cao, Mingsheng Long, Bin Liu, Jianmin Wang, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework (version 0.3.1)

Python 2.7/3.5

## Datasets
You can download the ImageNet dataset and NUS-WIDE dataset [here](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing).
As for COCO dataset, you can downloaded [here](http://mscoco.org/dataset/#download).
Each line in the list file follows the following format:
```
<image path><space><one hot label representation>
```

## Training
First, you can manually download the PyTorch pre-trained model introduced in `torchvision' library or if you have connected to the Internet, you can automatically downloaded them.
Then, you can train the model for each dataset using the followling command.
```
cd src
python train.py --gpu_id 0 --dataset coco --prefix resnet50_hashnet --hash_bit 48 --net ResNet50 --lr 0.0003 --class_num 1.0
```
## Evaluation
You can evaluate the Mean Average Precision(MAP) result on each dataset using the following command.
```
cd src
python test.py --gpu_id 0 --dataset coco --prefix resnet50_hashnet --hash_bit 48 --snapshot iter_09000
```