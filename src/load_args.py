"""
Command-line argument parsing.
"""

import argparse


def load_args():
    parser = argparse.ArgumentParser(description='Comparing Adam with different weight decay variants.')

    parser.add_argument('--optim-method', type=str, default='AdamProx',
                        choices=['AdamL2', 'AdamW', 'AdamProx'],
                        help='Which optimizer to use (default: AdamL2).')
    parser.add_argument('--eta0', type=float, default=0.001,
                        help='Initial learning rate (default: 0.1).')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005).')
    parser.add_argument('--scheduler', type=str, default='None',
                        choices=['Cosine', 'None'],
                        help='Which lr scheduler to use (default: None).')
    parser.add_argument('--loss-multiplier', type=float, default=1,
                        help='Multiply the loss by this factor, used for testing scale-freeness (default: 1).')

    # Training
    parser.add_argument('--train-epochs', type=int, default=100,
                        help='Number of train epochs (default: 100).')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='How many images in each train epoch (default: 128).')
    parser.add_argument('--validation', action='store_true',
                        help='Do validation (True) or test (False) (default: False).')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of training samples used as validation (default: 0.1).')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='How often should the model be evaluated during training, unit in epochs (default: 10).')
    parser.add_argument('--store-stats', action='store_true',
                        help='Store parameters, gradients, updates values during training (default: False).')
    parser.add_argument('--store-stats-interval', type=int, default=1,
                        help='How often should the stats of the model be stored each epoch during training, unit in iterations. (default: 1).')

    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset to run on (default: CIFAR10).')
    parser.add_argument('--dataroot', type=str, default='../data',
                        help='Where to retrieve data (default: ../data).')
    parser.add_argument('--model', type=str, default='ResNet20',
                        choices=['ResNet20', 'ResNet44', 'ResNet56',
                                 'ResNet110', 'ResNet218', 'DenseNetBC100'],
                        help='Which NN model to use (default: ResNet20).')
    parser.add_argument('--no-batch-norm', action='store_true',
                        help='Disable batch normalization (default: False).')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA (default: False).')
    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')

    return parser.parse_args()
