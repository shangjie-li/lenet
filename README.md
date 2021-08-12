# lenet

Implementation of LeNet in PyTorch for MNIST classification

## Installation
 - Install PyTorch environment with Anaconda
   ```
   conda create -n pytorch.v1.3.0 python=3.7
   conda activate pytorch.v1.3.0
   conda install pytorch==1.3.0 torchvision==0.4.1 cudatoolkit=1.0 -c pytorch
   pip install matplotlib
   ```
 - Clone this repository
   ```
   git clone https://github.com/shangjie-li/lenet.git
   ```

## Training
 - Run the training command below
   ```
   python train.py --lr=0.01 --epoch=10
   ```

## Evaluation
 - Run the evaluation command below
   ```
   python inference.py --weight_path=weights/2021-08-10-19-19.pth
   ```

## Test
 - Run the test command below
   ```
   python lenet_classifier.py
   ```
