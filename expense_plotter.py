
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import argparse
import re

def read_expense_report(path, header=4, skip_footer=7, clip_rows=3, clip_cols=1,
                        index_col=1):
    ext = os.path.splitext(path)[1]
    if ext == 'csv':
        f = pd.read_csv
    else:
        f = pd.read_excel
    r = f(path, header=header, skip_footer=skip_footer, parse_dates=True,
          index_col=index_col)
    r = r[clip_rows:]
    return r

def get_smoothed_amounts(path, binsize=30, smooth_window=60, yax='Amount'):
    """
    smooth_window : int
       The number of days to smooth over
    """
    r = read_expense_report(path)
    window = pd.offsets.Day(smooth_window)
    binsize = pd.offsets.Day(binsize)
    data = r[yax].resample(binsize).sum()
    sm = data.rolling(window, min_periods=1).mean()
    return sm

def plot_smoothed_amounts(paths, labels=None, binsize=30, smooth_window=60,
                          yax='Amount', ax=None, figsize=None, legend=True,
                          title=None):
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(1,1,1)
    for i, p in enumerate(paths):
        if labels is not None:
            label = labels[i]
        else:
            label = ''
        sm = get_smoothed_amounts(p, binsize=binsize,
                                  smooth_window=smooth_window, yax=yax)
        inds = sm.index.asi8
        inds = inds - inds[0]
        ax.plot(inds, sm.values, label=label)
    ax.set_xticks(inds)
    ax.set_xticklabels(sm.index.month)
    ax.set_ylabel('spending per {} days'.format(binsize))
    ax.set_xlabel('month')
    if legend and labels is not None:
        ax.legend(frameon=False)
    if title is not None:
        ax.set_title(title)
    return ax

def _clean_plot(ax, i, ticks=True, spines=True):
    if spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if i > 0:
        if spines:
            ax.spines['left'].set_visible(False)
        if ticks:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.yaxis.set_tick_params(size=0)

def generate_spending_summary(spending_path_dict, title_dict=None, labels=None,
                              binsize=30, smooth_window=60, yax='Amount',
                              main_plot='total', figsize=None):
    f = plt.figure(figsize=figsize)
    if main_plot in spending_path_dict.keys():
        other_plots = len(spending_path_dict.keys()) - 1
        spec = gs.GridSpec(2, other_plots)
        row = 1
        legend = False
        plot_keys = list(set(spending_path_dict.keys()).difference(main_plot))
        ax = f.add_subplot(spec[0, :])
        plot_smoothed_amounts(spending_path_dict[main_plot], labels, binsize,
                              smooth_window, yax, ax=ax, legend=True,
                              title=title_dict[main_plot])
        _clean_plot(ax, 0)
    else:
        num_plots = len(spending_path_dict.keys())
        spec = gs.GridSpec(1, num_plots)
        row = 0
        legend = True
        plot_keys = spending_path_dict.keys()
    for i, pk in enumerate(plot_keys):
        if i == 0:
            sharey_ax = None
            legend_p = legend
        else:
            sharey_ax = ax
            legend_p = False
        ax = f.add_subplot(spec[row, i], sharey=sharey_ax)
        paths = spending_path_dict[pk]
        title = title_dict[pk]
        plot_smoothed_amounts(paths, labels, binsize, smooth_window, yax, ax=ax,
                              legend=legend_p, title=title)
        if i > 0:
            ax.set_ylabel('')
        _clean_plot(ax, i)
    return f

def construct_path_dict(folder, category_list, title_list, year_list, ext='.xlsx',
                        file_template='{}_{}{}'):
    path_dict = {}
    title_dict = {}
    year_list = sorted(year_list)
    for i, cat in enumerate(category_list):
        file_list = []
        for year in year_list:
            file_string = file_template.format(cat, year, ext)
            full_string = os.path.join(folder, file_string)
            assert os.path.isfile(full_string)
            file_list.append(full_string)
        path_dict[cat] = file_list
        title_dict[cat] = title_list[i]
    return path_dict, title_dict, year_list

def make_parser():
    parser = argparse.ArgumentParser(description='script for plotting expense '
                                     'reports from Quickbooks')
    parser.add_argument('folder', help='folder containing expense reports generated '
                        'by Quickbooks', type=str)
    parser.add_argument('to_plot', type=str, nargs='*')
    parser.add_argument('-o', '--output', help='filename to save output plot in',
                        type=str, default='expense_plots')
    parser.add_argument('-t', '--output_filetype', help='filetype to save output as',
                        type=str, default='.pdf')
    parser.add_argument('-y', '--years', help='years to plot', nargs='*', type=str,
                        default=None)
    parser.add_argument('-n', '--names', help='long names of everything plotting, '
                        'given in the same order', nargs='*', type=str,
                        default=None)
    parser.add_argument('-e', '--extension', help='data filetype extension',
                        type=str, default='.xlsx')
    parser.add_argument('-c', '--column_title', default='Amount', type=str)
    parser.add_argument('-b', '--binsize', default=30, type=int)
    parser.add_argument('-s', '--smooth_window', default=60, type=int)
    parser.add_argument('-m', '--main_plot', default='total', type=str)
    parser.add_argument('-p', '--plot_size', default=(10, 3), type=int, nargs=2)
    return parser

def _get_years(folder):
    fls = os.listdir(folder)
    matches = [re.search('FY[0-9]{2}-[0-9]{2}', f) for f in fls]
    matches = filter(lambda x: x is not None, matches)
    matches = map(lambda x: x[0], matches)
    years = set(matches)
    years = list(years)
    return years

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.years is None:
        args.years = _get_years(args.folder)
    if args.names is None:
        args.names = args.to_plot
    assert len(args.names) == len(args.to_plot)
    out = construct_path_dict(args.folder, args.to_plot, args.names, args.years,
                              ext=args.extension)
    path_dict, title_dict, year_list = out
    outf = generate_spending_summary(path_dict, title_dict, year_list,
                                     binsize=args.binsize,
                                     smooth_window=args.smooth_window,
                                     yax=args.column_title,
                                     main_plot=args.main_plot,
                                     figsize=args.plot_size)
    savename = args.output + args.output_filetype
    outf.savefig(savename, bbox_inches='tight',
                 transparent=True)
