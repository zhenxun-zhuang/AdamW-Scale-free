# AdamW-Scale-free

This repo contains PyTorch codes for the experiments on image classification in the paper:

**[Understanding AdamW through Proximal Methods and Scale-freeness](https://arxiv.org/abs/2202.00089)**  
Zhenxun Zhuang, Mingrui Liu, Ashok Cutkosky, Francesco Orabona. Transactions on Machine Learning Research, 2022.

### Description

Adam has been widely adopted for training deep neural networks due to less hyperparameter tuning and remarkable performance. To improve generalization, Adam is typically used in tandem with a squared &ell;<sub>2</sub> regularizer (referred to as Adam-&ell;<sub>2</sub>). However, even better performance can be obtained with AdamW, which decouples the gradient of the regularizer from the update rule of Adam-&ell;<sub>2</sub>. Yet, we are still lacking a complete explanation of the advantages of AdamW. In this paper, we tackle this question from both an optimization and an empirical point of view. First, we show how to re-interpret AdamW as an approximation of a proximal gradient method, which takes advantage of the closed-form proximal mapping of the regularizer instead of only utilizing its gradient information as in Adam-&ell;<sub>2</sub>. Next, we consider the property of "scale-freeness" enjoyed by AdamW and by its proximal counterpart: their updates are invariant to component-wise rescaling of the gradients. We provide empirical evidence across a wide range of deep learning experiments showing a correlation between the problems in which AdamW exhibits an advantage over Adam-&ell;<sub>2</sub> and the degree to which we expect the gradients of the network to exhibit multiple scales, thus motivating the hypothesis that the advantage of AdamW could be due to the scale-free updates.

### Code & Usage

1. `src` folder contains codes for training a deep neural network to do image classification on CIFAR10/100. You can train models with the `main.py` script, with hyper-parameters being specified as flags (see `--help` for a detailed list and explanation).

2. `utils` folder contains codes for visualizing the results.

### Reproducing Results

#### ResNet on CIFAR10
```
python main.py --optim-method AdamL2 --eta0 0.001 --weight-decay 0.0005 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet20 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.001 --weight-decay 5e-05 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet20 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamL2 --eta0 0.0005 --weight-decay 0.0005 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet44 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.0005 --weight-decay 5e-05 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet44 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamL2 --eta0 0.0005 --weight-decay 0.0005 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet56 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.0005 --weight-decay 5e-05 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet56 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000`

python main.py --optim-method AdamL2 --eta0 0.001 --weight-decay 0.0005 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.0005 --weight-decay 0.0001 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamL2 --eta0 0.0005 --weight-decay 0.005 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet218 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.0005 --weight-decay 5e-05 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet218 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000


python main.py --optim-method AdamL2 --eta0 0.005 --weight-decay 0 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet110 --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.005 --weight-decay 0 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR10 --dataroot ../data --dataset CIFAR10 --model ResNet110 --scheduler None --store-stats --store-stats-interval 1000
```

#### DenseNet-BC 100 Layer on CIFAR100
```
python main.py --optim-method AdamL2 --eta0 0.0005 --weight-decay 0.001 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR100 --dataroot ../data --dataset CIFAR100 --model DenseNetBC100 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamW --eta0 0.001 --weight-decay 5e-5 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR100 --dataroot ../data --dataset CIFAR100 --model DenseNetBC100 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000

python main.py --optim-method AdamL2 --eta0 0.005 --weight-decay 0 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/CIFAR100 --dataroot ../data --dataset CIFAR100 --model DenseNetBC100 --scheduler None --store-stats --store-stats-interval 1000
```

#### Multiply the loss by a positive factor for checking scale-freeness.
```
python main.py --optim-method AdamL2 --eta0 0.001 --weight-decay 0.005 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/Loss_Mul --dataroot ../data --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000 --loss-multiplier 10

python main.py --optim-method AdamW --eta0 0.0005 --weight-decay 0.0001 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/Loss_Mul --dataroot ../data --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000 --loss-multiplier 10

python main.py --optim-method AdamL2 --eta0 0.0001 --weight-decay 0.1 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/Loss_Mul --dataroot ../data --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000 --loss-multiplier 100

python main.py --optim-method AdamW --eta0 0.0005 --weight-decay 0.0001 --train-epochs 300 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ../logs/Loss_Mul --dataroot ../data --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --store-stats --store-stats-interval 1000 --loss-multiplier 100
```
