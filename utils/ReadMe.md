### Scripts for visualizing results

Note: all figures are saved in `--fig-save-folder` without showing, please go there to check.

`draw_heatmap_tr_curve.py` draws the heatmap of final test error of `init_lr` vs. `weight_decay`, the training loss and the test accuracy curve of the best setting for different optimizers, for example:

```
python draw_heatmap_tr_curve.py --fig-save-folder ../figs --log-folder ../logs/CIFAR10 --dataset CIFAR10 --model ResNet110 --no-batch-norm --scheduler None --train-epochs 300 --batchsize 128 --loss-multiplier 1 --weight-decay-vals 0 1e-05 5e-05 0.0001 0.0005 0.001 0.005 0.01 --eta0-vals 5e-05 0.0001 0.0005 0.001 0.005 --optim-methods AdamL2 AdamW
```

`draw_histogram.py` draws the histogram of the magnitude of gradients, updates, parameters, etc., of the model in an epoch, for example:

```
python draw_histogram.py --stats-file-path ../logs/CIFAR10/CIFAR10_ResNet110_NoBN_AdamW_Eta0_0.0005_WD_0.0001_Scheduler_None_Loss_Mul_1_Epoch_300_BatchSize_128_Test.pickle --fig-save-folder ../figs --bin-power-base 2 --max-bin-value 1 --epoch-index 150
```
