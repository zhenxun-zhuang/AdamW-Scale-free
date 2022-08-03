import argparse
import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import entropy



GROUP_NAMES = ['parameter', 'gradient', 'update', 'update_no_alpha']
QUANTILE_LOCS = np.linspace(0, 1, 101, endpoint=True)
PLOT_FONTSIZE = 24
MAX_NUM_XTICKS = 10


def draw_fig(data, bin_base, min_bin_value, max_bin_value, optim_method, title, fig_save_folder='', epoch=0, no_xlabel=False, no_ylabel=False):
    plt.figure()
    x = list(range(len(data)))
    plt.bar(x, data)
    plt.ylim([0.0, 1.0])
    xlabels = list(range(min_bin_value, max_bin_value+1))
    if len(x) > MAX_NUM_XTICKS:
        ticks_to_show_ind = np.ceil(np.linspace(0, len(x) - 1, MAX_NUM_XTICKS)).astype('int')
        x_show = [x[i] for i in ticks_to_show_ind]
        x_labels_show = [xlabels[i] for i in ticks_to_show_ind]
        plt.xticks(x_show, labels=x_labels_show)
    else:
        plt.xticks(x, labels=xlabels)
    if not no_xlabel:
        plt.xlabel(f"Order of magnitude (power of {bin_base})", fontsize=PLOT_FONTSIZE)
    if not no_ylabel:
        plt.ylabel("Proportion", fontsize=PLOT_FONTSIZE)

    plt.tick_params(axis='both', which='major', labelsize=PLOT_FONTSIZE)
    plt.tick_params(axis='both', which='minor', labelsize=PLOT_FONTSIZE)
    plt.title(f"{optim_method} Epoch {epoch + 1}", fontsize=PLOT_FONTSIZE, loc='center') # ("\n".join(wrap(title, 40)))
    plt.tight_layout()
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder)
    plt.savefig(os.path.join(fig_save_folder, f"{title.replace(' ', '_')}_Epoch_{epoch}.png"),
                bbox_inches='tight')
    plt.close()


def log_base_n(number, base):
    return np.log(number) / np.log(base)


def bucketize(data_vector, bin_power_base, min_value, max_value):
    min_bin = np.floor(log_base_n(min_value, bin_power_base)).astype(int)
    max_bin = np.floor(log_base_n(max_value, bin_power_base)).astype(int)
    clamped_abs_data_vector = np.clip(np.abs(data_vector),
                                      min_value,
                                      max_value)
    order_of_magnitude = np.floor(log_base_n(clamped_abs_data_vector, bin_power_base)).astype(int)
    bins = [np.sum(order_of_magnitude == i) for i in range(min_bin, max_bin + 1)]
    return bins, min_bin, max_bin


# Note that we only care about the scale namely the absolute value here.
def merge_values_all_layers(name_per_layer, values_per_layer, eta0):
    all_values_per_group = {}
    for name, values_per_group in zip(name_per_layer, values_per_layer):
        if 'batchnorm' in name.lower():
            continue
        for group_name, values in values_per_group.items():
            if group_name not in all_values_per_group:
                all_values_per_group[group_name] = values.reshape(-1)
            else:
                all_values_per_group[group_name] = np.concatenate([all_values_per_group[group_name], values.reshape(-1)])
    all_values_per_group['update_no_alpha'] = all_values_per_group['update'] / eta0
    return all_values_per_group


def draw_histogram_main(args):
    with open(args.stats_file_path, 'rb') as f:
        cur_setting_stats = pickle.load(f)
    print(args.stats_file_path)
    setting_name = os.path.basename(args.stats_file_path)
    setting_name = setting_name[: setting_name.index('_Eta0')]
    optim_method = setting_name.split('_')[-1]
    eta0 = float(args.stats_file_path[args.stats_file_path.index('Eta0_') + 5: args.stats_file_path.index('_WD')])
    model_config = cur_setting_stats['layer names']
    target_stats = cur_setting_stats['model stats'][args.epoch_index][0]
    merged_values_all_layers = merge_values_all_layers(model_config, target_stats, eta0)
    for group_name, values_cur_group in merged_values_all_layers.items():
        abs_values_cur_group = np.abs(values_cur_group)
        if group_name not in GROUP_NAMES:
            continue
        bins, min_bin, max_bin = bucketize(abs_values_cur_group,
                                           bin_power_base=args.bin_power_base,
                                           min_value=args.min_bin_value,
                                           max_value=args.max_bin_value)
        bins_normalized = bins / np.sum(bins)
        fig_title = f"magnitude_histogram_{setting_name}_{group_name}"
        draw_fig(bins_normalized, args.bin_power_base, min_bin, max_bin, optim_method,
                 fig_title, fig_save_folder=os.path.join(args.fig_save_folder, f"{setting_name}_{args.epoch_index}"), epoch=args.epoch_index, no_xlabel=args.no_xlabel, no_ylabel=args.no_ylabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file-path", help="path of the stats files", type=str, default="CIFAR10/CIFAR10_resnet20_nobn_AdamW_Eta0_0.001_WD_5e-05_None_Epoch_300_Batch_128_Test.pickle")
    parser.add_argument("--fig-save-folder", help="folder to save figures", type=str, default='../figs')
    parser.add_argument("--bin-power-base", help="bin divided in what to the what power", type=int, default=2)
    parser.add_argument("--min-bin-value", help="minimum bin value", type=float, default=1e-8)
    parser.add_argument("--max-bin-value", help="maximum bin value", type=float, default=1e4)
    parser.add_argument("--epoch-index", help="which epoch to plot", type=int, default=299)
    parser.add_argument("--no-xlabel", help="do not plot xlabel", action='store_true')
    parser.add_argument("--no-ylabel", help="do not plot ylabel", action='store_true')
    args = parser.parse_args()
    draw_histogram_main(args)
