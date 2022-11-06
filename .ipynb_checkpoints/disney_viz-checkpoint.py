# Adapted from https://geoffruddock.com/matplotlib-experiment-visualizations/
import datetime as dt
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform

plt.rcParams['figure.facecolor'] = 'white'


def get_experiment_result_colors(
    interval_start: float, 
    interval_end: float
) -> Tuple[str, str]:
    
    """ Determine chart colors based on overlap of interval with zero. """
    if interval_start > 0:
        return 'darkseagreen', 'darkgreen'
    elif interval_end < 0:
        return 'darksalmon', 'darkred'
    else:
        return 'lightgray', 'gray'


def plot_experiment_single_group(
    ax, 
    sub_df: pd.DataFrame
) -> Tuple[float, float]:
    
    """ Plot each row of a DataFrame on the same mpl axis object. """

    ytick_labels = []
    x_min, x_max = 0, 0

    # Iterate over each row in group, reversing order since mpl plots from bottom up
    for j, (dim, row) in enumerate(sub_df.iloc[::-1].iterrows()):
        if isinstance(dim, tuple):
            dim = dim[1]

        # Calculate z-score for each test based on test-specific correction factor
        z = norm(0, 1).ppf(1 - row.alpha / 2)
        interval_start = row.uplift - (z * row.std_err)
        interval_end = row.uplift + (z * row.std_err)

        # Conditional coloring based on significance of result
        fill_color, edge_color = get_experiment_result_colors(interval_start, interval_end)

        ax.barh(j, [z * row.std_err, z * row.std_err],
                left=[interval_start, interval_start + z * row.std_err],
                height=0.8,
                color=fill_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=3)

        ytick_labels.append(dim)
        x_min = min(x_min, interval_start - 0.01)
        x_max = max(x_max, interval_end + 0.01)

    # Axis-specific formatting

    ax.xaxis.grid(True, alpha=0.4)
    ax.xaxis.set_ticks_position('none')
    ax.axvline(0.00, color='black', linewidth=1.1, zorder=2)
    ax.yaxis.tick_right()
    ax.set_yticks(np.arange(len(sub_df)))
    ax.set_yticklabels(ytick_labels)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min-0.4, y_max+0.4)
    ax.yaxis.set_ticks_position('none')

    return x_min, x_max


def plot_experiment_results(df: pd.DataFrame, title: str = None, sample_size: int = None, combine_axes: bool = False) -> None:
    """ Plot a (possibly MultiIndex) DataFrame on one or more matplotlib axes.
    Args:
        df (pd.DataFrame): DataFrame with MultiIndex representing dimensions or KPIs, and following cols: uplift, std_err, alpha
        title (str): Title displayed above plot
        sample_size (int): Used to add contextual information to bottom corner of plot
        combine_axes (bool): If true and input df has multiindex, collapse axes together into one visible axis.
    """

    plt.rcParams['figure.facecolor'] = 'white'

    n_levels = len(df.index.names)
    if n_levels > 2:
        raise ValueError
    elif n_levels == 2:
        plt_rows = df.index.get_level_values(0).nunique()
    else:
        plt_rows = 1

    # Make an axis for each group of MultiIndex DataFrame input
    fig, axes = plt.subplots(nrows=plt_rows,
                             ncols=1,
                             sharex=True,
                             figsize=(6, 0.5 * df.shape[0] + 0.2), dpi=100)

    if n_levels == 1:
        ax = axes
        x_min, x_max = plot_experiment_single_group(ax, df)

    if n_levels == 2:
        # Iterate over top-level groupings of index
        x_mins, x_maxs = [], []
        for i, (group, results) in enumerate(df.groupby(level=0, sort=False)):
            ax = axes[i]
            a, b = plot_experiment_single_group(ax, results)
            x_mins.append(a)
            x_maxs.append(b)
            ax.set_ylabel(group)

        x_min = min(x_mins)
        x_max = max(x_maxs)
        ax = axes[-1]  # set variable back to final axis for downstream formatting functions

        if combine_axes:
            fig.subplots_adjust(hspace=0)
            axes[0].spines['bottom'].set_visible(False)
            axes[-1].spines['top'].set_visible(False)
            for axis in axes[1:-1]:
                axis.spines['bottom'].set_visible(False)
                axis.spines['top'].set_visible(False)

    ax.set_xlim(x_min, x_max)
    x_tick_width = (1 + np.floor((x_max - x_min)/0.10)) / 100
    loc = plticker.MultipleLocator(base=x_tick_width)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.set_xticklabels(['{:.0%}'.format(x) for x in ax.get_xticks()])
    ax.set_xlabel('Uplift (relative)')

    # Add title, sample size, and timestamp labels to plot
    fig.text(0.5, 0.95 - 0.025 * n_levels, title, size='x-large', horizontalalignment='center')

    vertical_offset = - (0.1 + 0.2 * n_levels)
    # timestamp_text = dt.datetime.now().strftime('Analyzed: %Y-%m-%d')
    # fig.text(1, vertical_offset,
    #          timestamp_text,
    #          size='small', color='grey',
    #          ha='right', wrap=True, transform=ax.transAxes)

    if sample_size:
        sample_size_text = f'Sample size: {int(sample_size/1000)}K'
        fig.text(0, vertical_offset,
                 sample_size_text,
                 size='small', color='grey',
                 ha='left', wrap=True, transform=ax.transAxes)
        
    return fig, ax


def stacked_bar(
    data,
    x,
    y,
    hue,
    cmap='Accent',
    normalize=True,
    group_colors=None,
    barWidth=0.85,
    xtick_rotation=0,
    x_order=None,
    hue_order=None,
):
    """
    Creates a stacked bar chart.  Input dataframe should not be pivoted.  
    
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe should not be pivoted.

    x : str
        Column name that will serve as the x ticks for the stacked bar chart

    y : str
        Column name that contains the values to be plotted on the y axis

    hue : str
        Column name that contains the groups that will be stacked

    cmap : str
        Name of the seaborn color map for choosing colors.  Default is "Accent",
        other popular cmaps include the seaborn default 'tab10', and 'Set2'.

    normalize : boolean
        If true, then the each stacked bar will sum to 1.

    group_colors : dictionary
        Dictionary map of each hue group to its color

    barWidth : float   
        Width of each bar

    xtick_rotation : float
        Degrees of rotation fot he xtick labels

    x_order : list
        Order for the x-axis 

    hue_order : list
        Stacking order (bottom to top) of hue groups for each bar  

    Returns
    -------
    fig : matplotlib.pyplot.figure

    ax : matplotlib.pyplot.ax

    group_colors : dictionary
        The mapping of each hue group to its color
    """

    # Pivot the data 
    dfp = data.pivot(index=x, columns=hue, values=y)
    
    # Sort the order of the x-axis and get labels
    if x_order == None:
        dfp.sort_values(x, inplace=True)
    else:
        dfp = dfp.loc[x_order]
    xtick_labels = dfp.index.to_list()
    
    # If normalize=True, divide each row by the row total
    if normalize==True:
        row_totals = dfp.sum(axis=1)
        dfp = dfp.div(row_totals, axis=0)
    
    # Fill missing values with 0
    dfp.fillna(0, inplace=True)
    
    # Get the order of the groups by largest share on bottom
    if hue_order==None:
        groups = dfp.sum().sort_values(ascending=False).index.to_list()
    else:
        groups = hue_order
    
    # Get group colors
    if group_colors==None:
        palette = sns.color_palette(cmap, len(groups))
        group_colors = {groups[i]: palette[i] for i in range(len(groups))}

        # If np.nan present, make a string version of the key
        # for proper legend handling 
        if np.nan in group_colors.keys():
            group_colors['nan'] = group_colors[np.nan]
    
    # Get xtick_spacings
    xtick_positions = range(len(dfp))
    bottoms = np.zeros(len(dfp))

    # Make bar plots
    fig, ax = plt.subplots(figsize=(len(xtick_labels), 5))
    for group in groups:
        ax.bar(
            xtick_positions, 
            dfp[group], 
            bottom=bottoms, 
            edgecolor='white', 
            width=barWidth, 
            color=group_colors[group], 
            label=group
        )
        bottoms = bottoms + dfp[group]

    # Reverse the order of legend to match bar ordering
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], title=hue, bbox_to_anchor=(1, 1), loc='upper left')

    # Color the legend label text
    for text in leg.get_texts():
        leg_label = text.get_text()
        plt.setp(text, color=group_colors[leg_label], fontweight='semibold', fontsize=12)

    plt.xticks(xtick_positions, xtick_labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=xtick_rotation)
    return fig, ax, group_colors

