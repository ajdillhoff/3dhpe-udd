import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from matplotlib import cm

from utils.util import get_errors

FONT_SIZE_XLABEL = 15
FONT_SIZE_YLABEL = 15
FONT_SIZE_LEGEND = 11.8
FONT_SIZE_TICK = 11.8


def get_comp():
    files = ["results/comp/CVWW15_NYU_Prior.txt",
             "results/comp/ICCV15_NYU_Feedback.txt",
             "results/comp/ICCVW17_NYU_DeepPrior++.txt",
             "results/comp/IJCAI16_NYU_DeepModel.txt",
             "results/comp/WACV19_NYU_murauer_n72757_uvd.txt"]
    errs = []
    names = ["DeepPrior",
             "Feedback",
             "DeepPrior++",
             "DeepModel",
             "MURAUER"]
    for file in files:
        err = get_errors('nyu', file)
        errs.append(err)
    return errs, names


def plot_dibra(ax):
    x_rep = np.array([0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0])
    y_rep = np.array([0.0, 0.0, 0.04, 0.216, 0.43, 0.612, 0.75, 0.836, 0.866])
    y_rep *= 100
    x_vals = np.linspace(0, 75.0, num=1000)
    f = interp1d(x_rep, y_rep, kind='cubic')
    y_int = f(x_vals)
    # interp yields non-zero values in the beginning
    y_int[:int(1000/len(y_rep))] = 0.0
    ax.plot(x_vals, y_int, c='b', label='Dibra (Base)')


def draw_error_curve(errs, eval_names, metric_type, fig):
    eval_num = len(errs)
    thresholds = np.arange(0, 85, 1)
    results = np.zeros(thresholds.shape+(eval_num,))
    #fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    xlabel = 'Mean distance threshold (mm)'
    ylabel = 'Fraction of frames within distance (%)'
    # color map
    jet = plt.get_cmap('jet')
    values = range(eval_num)
    if eval_num < 3:
          jet = plt.get_cmap('prism')
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    l_styles = ['-','--']

    for eval_idx in range(eval_num):
        if metric_type == 'mean-frame':
            err = np.mean(errs[eval_idx], axis=1)
        elif  metric_type == 'max-frame':
            err = np.max(errs[eval_idx], axis=1)
            xlabel = 'Maximum allowed distance to GT (mm)'
        elif  metric_type == 'joint':
            err = errs[eval_idx]
            xlabel = 'Distance Threshold (mm)'
            ylabel = 'Fraction of joints within distance (%)'
        err_flat = err.ravel()
        for idx, th in enumerate(thresholds):
            results[idx, eval_idx] = np.where(err_flat <= th)[0].shape[0] * 1.0 / err_flat.shape[0]
        colorVal = scalarMap.to_rgba(eval_idx)
        ls = l_styles[eval_idx%len(l_styles)]
        if eval_idx == eval_num - 1:
            ls = '-'
        ax.plot(thresholds, results[:, eval_idx]*100, label=eval_names[eval_idx],
                color=colorVal, linestyle=ls)

    # Plot competing methods
    plot_dibra(ax)

    plt.xlabel(xlabel, fontsize=FONT_SIZE_XLABEL)
    plt.ylabel(ylabel, fontsize=FONT_SIZE_YLABEL)
    ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    plt.grid(True)
    major_ticks = np.arange(0, 81, 10)
    minor_ticks = np.arange(0, 81, 5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    major_ticks = np.arange(0, 101, 10)
    minor_ticks = np.arange(0, 101, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.3)
    ax.grid(which='major', alpha=0.5, linestyle='--', linewidth=0.3)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelsize=FONT_SIZE_TICK)
    fig.tight_layout()


def main():
    eval_names = []
    eval_files = []
    eval_errs = []
    for idx in range(1, len(sys.argv), 2):
        in_name = sys.argv[idx]
        in_file = sys.argv[idx+1]
        eval_names.append(in_name)
        eval_files.append(in_file)
        err = np.load(in_file)
        eval_errs.append(err)
    comp_results, comp_names = get_comp()
    eval_errs += comp_results
    eval_names += comp_names

    fig = plt.figure(figsize=(8, 6))
    plt.figure(fig.number)
    draw_error_curve(eval_errs, eval_names, 'max-frame', fig)

    plt.show()


if __name__ == '__main__':
    main()

