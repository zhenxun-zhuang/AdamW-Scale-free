import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os


SAVE_FIG = True
GOOD_COLORS = [
    [0, 0, 1.0000],
    [1.0000, 0, 0],
    [0.1, 0.9500, 0.1],
    [0, 0, 0.1724],
    [1.0000, 0.1034, 0.7241],
    [1.0000, 0.8276, 0],
    [0.7241, 0.3103, 0.8276],
    [0.5172, 0.5172, 1.0000]
]
OPTIM_METHOD_COLOR_INC = {'AdamL2': 0,
                          'AdamW': 1,
                          'AdamProx': 2}


def draw_tr_curve(best_configs, num_epochs, optim_methods, log_folder, fig_save_folder, tr_ylim, ts_ylim, log_name_prefix, log_name_suffix):
    _, axs = plt.subplots(1, 2, figsize=[20, 6])
    linewidth = 4
    fontsize = 20
    x = list(range(1, num_epochs + 1))

    for optim_method in optim_methods:
        color = GOOD_COLORS[OPTIM_METHOD_COLOR_INC.get(optim_method, 4)]
        eta0 = best_configs[optim_method]["eta0"]
        wd = best_configs[optim_method]["wd"]
        filename = f"{log_name_prefix}_{optim_method}_Eta0_{eta0}_WD_{wd}_{log_name_suffix}"
        fullFileName = os.path.join(log_folder, filename)
        if os.path.isfile(fullFileName):
            with open(fullFileName, 'r') as f:
                stat_names = ['Training Loss', "Training Accuracy",
                              'Test Loss', 'Test Accuracy']
                results = {}
                counter = 0
                for line in f:
                    if not line.startswith('['):
                        continue
                    stat_name = stat_names[counter]
                    results[stat_name] = np.array(eval(line))
                    counter = (counter + 1) % 4
                axs[0].plot(x, results['Training Loss'], label=optim_method, color=color, LineWidth=linewidth)
                axs[1].plot(x, results['Test Accuracy'], label=optim_method, color=color, LineWidth=linewidth)
                print(f"{optim_method}: eta0--{eta0}, weight decay--{wd}")
                print(f"Final training loss {'%.4f' % results['Training Loss'][-1]}")
                print(f"Final test accuracy {'%.4f' % results['Test Accuracy'][-1]}")
                print("\n")
    axs[0].set_xlabel('Epoch', fontsize=fontsize)
    axs[0].set_ylabel('Training Loss', fontsize=fontsize)
    axs[0].set(ylim=tr_ylim)
    axs[0].tick_params(axis='both', which='major', labelsize=fontsize)
    axs[0].tick_params(axis='both', which='minor', labelsize=fontsize)
    axs[0].legend(fontsize=fontsize)
    axs[1].set_xlabel('Epoch', fontsize=fontsize)
    axs[1].set_ylabel('Test Accuracy', fontsize=fontsize)
    axs[1].set(ylim=ts_ylim)
    axs[1].tick_params(axis='both', which='major', labelsize=fontsize)
    axs[1].tick_params(axis='both', which='minor', labelsize=fontsize)
    axs[1].legend(fontsize=fontsize)
    if not SAVE_FIG:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    else:
        plt.savefig(os.path.join(fig_save_folder, f"{log_name_prefix}.png"),
                    bbox_inches='tight')
    plt.close()


def heatmap2d(optim_methods, test_errors, xticks, yticks, heatmap_max_threshold, heatmap_min_threshold, fig_save_folder, fig_save_name):
    num_methods = len(optim_methods)
    fig, axs = plt.subplots(1, num_methods, figsize=[20, 8])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fontsize = 18

    best_configs = {}
    data_name = 'test error'
    for col, optim_method in enumerate(optim_methods):
        data = test_errors[optim_method]
        best_lr_idx, best_wd_idx = np.unravel_index(data.argmin(), data.shape)
        pcm = axs[col].imshow(data, cmap='jet', interpolation="bilinear",
                              vmin=heatmap_min_threshold,
                              vmax=heatmap_max_threshold)
        best_setting_patch = Circle((best_wd_idx, best_lr_idx),
                                    radius=0.2, color='black')
        axs[col].add_patch(best_setting_patch)
        axs[col].set_xticks(np.arange(len(data[0])))
        axs[col].set_xticklabels(xticks[:len(data[0])])
        axs[col].set_yticks(np.arange(len(yticks)))
        axs[col].set_yticklabels(yticks)
        axs[col].tick_params(axis='both', which='major', labelsize=fontsize-2)
        axs[col].tick_params(axis='both', which='minor', labelsize=fontsize-2)
        axs[col].set_xlabel('Weight decay', fontsize=fontsize)
        axs[col].set_ylabel('Initial stepsize', fontsize=fontsize)
        axs[col].set_title(f"{optim_method}--Best {data_name}: {'%.4f' % data.min()}", fontsize=fontsize)
        cbar = fig.colorbar(pcm, ax=axs[col], fraction=0.04, pad=0.04)
        cbar.ax.tick_params(labelsize=fontsize)
        print(optim_method)
        print(f'Final {data_name}:')
        with np.printoptions(precision=4, suppress=True):
            print(data)
        print(f"Best: eta0--{yticks[best_lr_idx]}, weight decay--{xticks[best_wd_idx]}, {data_name}--{data.min()}\n\n")
        best_configs[optim_method] = {"eta0": yticks[best_lr_idx], "wd": xticks[best_wd_idx]}

    if not SAVE_FIG:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    else:
        plt.savefig(os.path.join(fig_save_folder, f"{fig_save_name}.png"), bbox_inches='tight')
    plt.close()

    return best_configs


def main(args):
    log_name_prefix = (f'{args.dataset}_{args.model}_'
                       + ('NoBN' if args.no_batch_norm else 'BN'))
    log_name_suffix = ((('Scheduler_%s_' % args.scheduler) if args.scheduler else '')
                       + ('Loss_Mul_%g_' % args.loss_multiplier)
                       + ('Epoch_%d_BatchSize_%d_' % (args.train_epochs, args.batchsize))
                       + ('%s' % ('Validation' if args.validation else 'Test'))
                       + '.txt')

    if args.dataset == 'CIFAR10':
        heatmap_min_threshold = 0
        heatmap_max_threshold = 0.4
        tr_ylim = [0, 1]
        ts_ylim = [0.6, 1.0]
    elif args.dataset == 'CIFAR100':
        heatmap_min_threshold = 0.3
        heatmap_max_threshold = 0.6
        tr_ylim = [0, 1]
        ts_ylim = [0.4, 0.7]

    test_errors = {optim_method:np.ones((len(args.eta0_vals), len(args.weight_decay_vals)))
                   for optim_method in args.optim_methods}
    for optim_method in args.optim_methods:
        for column, weight_decay in enumerate(args.weight_decay_vals):
            for row, eta0 in enumerate(args.eta0_vals):
                filename = f"{log_name_prefix}_{optim_method}_Eta0_{eta0}_WD_{weight_decay}_{log_name_suffix}"
                fullFileName = os.path.join(args.log_folder, filename)
                if os.path.isfile(fullFileName):
                    with open(fullFileName, 'r') as f:
                        all_lines = f.readlines()
                        cur_tr_loss = all_lines[4][len("Final training loss is "):].strip()
                        if cur_tr_loss == 'nan':
                            test_errors[optim_method][row][column] = 1.1
                        else:
                            test_errors[optim_method][row][column] = 1.0 - float(all_lines[11][len("Final test accuracy is "):])
    print(log_name_prefix + '\n')
    if not os.path.exists(args.fig_save_folder):
        os.makedirs(args.fig_save_folder)
    best_configs = heatmap2d(optim_methods=args.optim_methods,
                             test_errors=test_errors,
                             xticks=args.weight_decay_vals,
                             yticks=args.eta0_vals,
                             heatmap_max_threshold=heatmap_max_threshold,
                             heatmap_min_threshold=heatmap_min_threshold,
                             fig_save_folder=args.fig_save_folder,
                             fig_save_name=log_name_prefix + '_heatmap')
    draw_tr_curve(best_configs=best_configs,
                  num_epochs=args.train_epochs,
                  optim_methods=args.optim_methods,
                  log_folder=args.log_folder,
                  fig_save_folder=args.fig_save_folder,
                  tr_ylim=tr_ylim,
                  ts_ylim=ts_ylim,
                  log_name_prefix=log_name_prefix,
                  log_name_suffix=log_name_suffix)


parser = argparse.ArgumentParser()
parser.add_argument("--fig-save-folder", help="folder to save figures", type=str, default='figs')
parser.add_argument("--log-folder", help="folder to read logs", type=str, default='logs')
parser.add_argument("--train-epochs", help="how many epochs trained", type=int, default=300)
parser.add_argument('--batchsize', type=int, default=128,
                    help='How many images in each train epoch (default: 128).')
parser.add_argument('--validation', action='store_true',
                    help='Do validation (True) or test (False) (default: False).')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'CIFAR100'],
                    help='Which dataset to run on (default: CIFAR10).')
parser.add_argument('--model', type=str, default='ResNet20',
                    choices=['ResNet20', 'ResNet44', 'ResNet56',
                             'ResNet110', 'ResNet218', 'DenseNetBC100'],
                    help='Which NN model to use (default: ResNet20).')
parser.add_argument('--no-batch-norm', action='store_true',
                    help='Disable batch normalization (default: False).')
parser.add_argument('--scheduler', type=str, default='None',
                    choices=['Cosine', 'None'],
                    help='Which lr scheduler to use (default: None).')
parser.add_argument('--loss-multiplier', type=float, default=1,
                    help='Multiply the loss by this factor, used for testing scale-freeness (default: 1).')
parser.add_argument('--optim-methods', nargs='*',
                    help='Type of optimizers to draw.')
parser.add_argument('--eta0-vals', nargs='*',
                    help='Values of initial step sizes you want to draw.')
parser.add_argument('--weight-decay-vals', nargs='*',
                    help='Values of weight decay you want to draw.')
args = parser.parse_args()
main(args)
