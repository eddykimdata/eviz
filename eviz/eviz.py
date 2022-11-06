# Adapted from https://geoffruddock.com/matplotlib-experiment-visualizations/
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re
import os

plt.rcParams['figure.facecolor'] = 'white'

# logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def shares_plot(
    data,
    x,
    y,
    hue,
    kind='line',
    cmap='tab10',
    normalize=True,
    group_colors=None,
    barWidth=0.85,
    xtick_rotation=0,
    x_order=None,
    hue_order=None,
    marker=None,
    markersize=6,
    grid_height=100,
    margin_plot_height=20,
    margin_plot_spacer=3,
    plot_totals=None,
    totals_label='Total',
    normalize_totals=True,
    totals_marker=None,
    totals_markersize=6,
    height=6,
    width=None,
    reverse_legend_order=True
):
    """
    Creates a plot (either stacked bar or line plot) describing the fraction of shares 
    from each given bucket class.  Example: x-dimension is DS, and we want to know for
    each DS, the number (or portion) of videos that were watched that were classified as 
    'series', 'movie', 'short-form', or 'other'.  The input dataframe should not be
    pivoted, and should be in the form:

    |        DS       | CONTENT_CLASS   |    N |
    |----------------:|:----------------|-----:|
    |      2020-09-01 | other           |  505 |
    |      2020-09-01 | movie           |  185 |
    |      2020-09-01 | short-form      |   12 |
    |      2020-09-01 | series          | 2135 |
    |      2020-09-02 | short-form      |   50 |
    |      2020-09-02 | other           |  713 |

    
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
    
    marker : str
        Only valid for line plot, add a marker shape

    markersize : int
        Only valid for line plot, changes size of marker

    grid_height : int
        Only valid if plot_totals is not None, this number will determine the 
        relative height of the two subplots (main plot height + totals plot height)

    margin_plot_height : int
        Only valid if plot_totals is not None, this number will determine the size 
        of the margin plot, with the remainder of the grid_height being 
        allocated to the main plot's height and margin_plot_spacer.  

    margin_plot_spacer : int
        Only valid if plot_totals is not None, this number will create additional 
        space between the two subplots, allocated from the total grid_height
        (the main plot's height will be grid_height - margin_plot_height - 
        margin_plot_spacer). 

    plot_totals : str
        Default value: None.  Parameter value can be [None, 'bar', 'line'].  
        If not None, then a subplot will plot the totals as a bar or line plot.

    totals_label : str
        Label for the totals subplot legend

    normalize_totals : boolean
        Default value: True.  If True, then the totals subplot will be normalized
        such that the sum of all x-value totals will equal 1

    totals_marker : str
        Default: None.  Only valid if plot_totals == 'line'. Adds a marker shape
        to the totals subplot.  

    totals_markersize : int
        Default: 6.  Only valid if plot_totals == 'line'. Adjusts themarker size
        for the totals subplot.  

    height : int
        Default: 6.  Height of the figure in inches

    width : int,
        Default: None.  Width of the figure in inches.  If None, then width
        is set such that aspect ratio width:height is 4:3.  

    reverse_legend_order : boolean
        Default: True.  Changes the order of labels in the legend.  If True,
        top label will be the line or bar with the greatest total value (i.e.
        corresponds such that top legend label will be the top bar in stacked
        bar, or the top line in lineplot).

    Returns
    -------
    fig : matplotlib.pyplot.figure

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

    # Get the row totals 
    totals = dfp.sum(axis=1)

    # If normalize=True, divide each row by the row total
    if normalize==True:
        dfp = dfp.div(totals, axis=0)
    
    
    # Get group names, order by largest share first
    if hue_order==None:
        groups = dfp.sum().sort_values(ascending=False).index.to_list()
    else:
        groups = hue_order

    # Get group colors
    if group_colors==None:
        group_colors = get_plot_colors(groups, cmap)

    # Get default figure size
    if width==None:
        if kind=='bar':
            width = len(dfp)
        else:
            width = height / 0.75

    

    if plot_totals:
        if plot_totals not in ['bar', 'line']:
            raise ValueError("plot_totals must be in ['bar', 'line']")
        
        fig, ax_main, ax_marg = get_dual_gridspec(
            height=height,
            width=width,
            grid_height=grid_height,
            margin_plot_height=margin_plot_height,
            margin_plot_spacer=margin_plot_spacer)

        # fig = plt.figure(figsize=(width, height))

        # gs = plt.GridSpec(grid_height, grid_height)
        # ax_main = fig.add_subplot(gs[margin_plot_height+margin_plot_spacer:, :-1])
        # ax_marg = fig.add_subplot(gs[0:margin_plot_height, :-1], sharex=ax_main)

        # # Turn off tick visibility for the measure axis on the marginal plots
        # plt.setp(ax_marg.get_xticklabels(), visible=False)
        # plt.setp(ax_marg.get_xticklabels(minor=True), visible=False)
        # plt.setp(ax_marg.xaxis.get_majorticklines(), visible=False)
        # plt.setp(ax_marg.xaxis.get_minorticklines(), visible=False)

        # Plot totals
        ax_marg = make_totals_plot(
            totals=totals, 
            plot_totals=plot_totals, 
            normalize_totals=normalize_totals, 
            totals_label=totals_label, 
            totals_marker=totals_marker,
            totals_markersize=totals_markersize,
            ax=ax_marg)

    else:
        fig, ax_main = plt.subplots(figsize=(width, height))

    if kind=='bar':
        ax_main = stacked_bar(
            df=dfp,
            x=x,
            y=y,
            hue=hue,
            groups=groups,
            barWidth=barWidth,
            group_colors=group_colors,
            xtick_rotation=xtick_rotation,
            ax=ax_main
        )

    elif kind=='line':
        ax_main = lineplot(
            df=dfp.reset_index(),
            x=x,
            y=y,
            hue=hue,
            cols=groups,
            col_colors=group_colors,
            xtick_rotation=xtick_rotation,
            marker=marker,
            markersize=markersize,
            ax=ax_main
        )

    else:
        raise ValueError("kind must be in ['line', 'bar']")

    return fig, group_colors


def stacked_bar(
    df,
    x,
    y,
    hue,
    groups,
    barWidth,
    group_colors,
    xtick_rotation,
    ax,
    reverse_legend_order=True
):
    """
    Creates a stacked bar plot.  Called from the shares_plot function.  
    See docstring for shares_plot.
    """

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Get xtick_spacings and y bottoms
    if df.index.dtype==str:
        xtick_positions = range(len(df))
    else:
        xtick_positions = df.index

    xtick_labels = df.index.to_list()

    bottoms = np.zeros(len(df))

    # Make bar plots
    for group in groups:
        ax.bar(
            xtick_positions, 
            df[group], 
            bottom=bottoms, 
            edgecolor='white', 
            width=barWidth, 
            color=group_colors[group], 
            label=group
        )
        bottoms = bottoms + df[group]

    # Reverse the order of legend to match bar ordering
    leg = get_legend(ax, reverse_legend_order=reverse_legend_order)

    plt.xticks(xtick_positions, xtick_labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(rotation=xtick_rotation)

    return ax


def lineplot(
    df,
    x,
    y,
    hue,
    cols,
    col_colors,
    xtick_rotation,
    marker,
    markersize,
    ax,
    reverse_legend_order=True
):
    """
    See docstring for shares_plot
    """

    # Make line plots
    for col in cols:
        ax.plot(
            df[x], 
            df[col], 
            color=col_colors[col], 
            label=col,
            marker=marker,
            markersize=markersize
        )

    leg = get_legend(ax, reverse_legend_order=reverse_legend_order)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(rotation=xtick_rotation)

    return ax


def make_totals_plot(
    totals, 
    plot_totals, 
    normalize_totals, 
    totals_label, 
    totals_marker,
    totals_markersize,
    ax):
    
    if normalize_totals==True:
        totals = totals / totals.sum()

    if plot_totals=='line':
        ax.plot(
            totals.index, 
            totals,
            marker=totals_marker, 
            markersize=totals_markersize, 
            label=totals_label)
    elif plot_totals=='bar':
        ax.bar(totals.index, totals, label=totals_label)

    leg = ax.legend(title='TOTAL', bbox_to_anchor=(1, 1), loc='upper left')

    return ax


def get_legend(ax, title=None, reverse_legend_order=True, bbox_to_anchor=(1,1), loc='upper left'):
    """
    Returns a legend to the right side of the plot, with legend text colors bolded
    and matching the plot colors
    """
    handles, labels = ax.get_legend_handles_labels()
    
    if reverse_legend_order==True:
        handles = handles[::-1]
        labels = labels[::-1]

    leg = ax.legend(handles, labels, title=title, bbox_to_anchor=bbox_to_anchor, loc=loc)

    for handle, text in zip(leg.legendHandles, leg.get_texts()):
        text.set_fontweight('semibold')
        text.set_fontsize(12)
        
        # Set text color to equal symbol color.  Different method for line vs patch
        if hasattr(handle, 'get_facecolor'):
            text_color = handle.get_facecolor()
            # If alpha transparency is applied, grab the original color
            text_color = text_color[0:3]
        elif hasattr(handle, 'get_color'):
            text_color = handle.get_color()
        else:
            text_color = 'black'
        
        if type(text_color)==np.ndarray:
            text_color = text_color.flatten()
        text.set_color(text_color)

    return leg


def get_plot_colors(
    groups,
    cmap='tab10',
):
    # Get group colors
    palette = sns.color_palette(cmap, len(groups))
    group_colors = {groups[i]: palette[i] for i in range(len(groups))}

    # If np.nan present, make a string version of the key
    # for proper legend handling 
    if np.nan in group_colors.keys():
        group_colors['nan'] = group_colors[np.nan]

    return group_colors

def get_dual_gridspec(
    height=6,
    width=8,
    grid_height=100,
    margin_plot_height=20,
    margin_plot_spacer=3,
):
    """Creates a dual axes plot that shares the x axis.

    Parameters
    ----------
    height : int
        Height of the figure in inches

    width : int,
        Width of the figure in inches. 

    grid_height : int
        Only valid if plot_totals is not None, this number will determine the 
        relative height of the two subplots (main plot height + totals plot height)

    margin_plot_height : int
        Only valid if plot_totals is not None, this number will determine the size 
        of the margin plot, with the remainder of the grid_height being 
        allocated to the main plot's height and margin_plot_spacer.  

    margin_plot_spacer : int
        Only valid if plot_totals is not None, this number will create additional 
        space between the two subplots, allocated from the total grid_height
        (the main plot's height will be grid_height - margin_plot_height - 
        margin_plot_spacer). 
    """
    fig = plt.figure(figsize=(width, height))
    gs = plt.GridSpec(grid_height, grid_height)
    ax_main = fig.add_subplot(gs[margin_plot_height+margin_plot_spacer:, :-1])
    ax_marg = fig.add_subplot(gs[0:margin_plot_height, :-1], sharex=ax_main)


    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg.get_xticklabels(), visible=False)
    plt.setp(ax_marg.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg.xaxis.get_minorticklines(), visible=False)

    return fig, ax_main, ax_marg


def remove_outliers(
    df, 
    outlier_col,
    lower_quantile=0.01,
    upper_quantile=0.99
):
    """Remove outliers for plotting histograms"""

    # To guard against features that are bounded by 0 and 1, we don't want 
    # to remove values 0 and 1 even if they are outliers.  

    lower_bound = df[outlier_col].quantile(lower_quantile)-1
    upper_bound = df[outlier_col].quantile(upper_quantile)+1

    return df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]

def hist_mirrored(
    df,
    target_col,
    hue=None,
    width=6,
    height=4,
    hue_order=None,
    color_top='#619CFF',
    color_bottom='#F8766D',
    ignore_outliers=True,
    **hist_kwargs
):
    """
    For data with a binary split, this function plots a mirrored histogram that 
    shares the same x-axis.  
    
    Parameters
    ----------
    df : pd.DataFrame

    target_col : str 
        The column name that contains the values to be plotted.

    hue : str
        The column name that contains the binary split values.  This column must
        contain exactly two unique values.

    width : int,
        Width of the figure in inches. 

    height : int
        Height of the figure in inches

    hue_order : list
        If given, the top plot will contain the histogram where the hue value equals
        the first item in hue_order, the bottom plot will contain the histogram 
        where the hue value equals the last item in hue_order.

    color_top : str
        Color for the histogram bars for top histogram

    color_bottom : str
        Color for the histogram bars for bottom histogram

    ignore_outliers : boolean
        If True, the histogram will only consider values between 1st percentile
        and 99th percentile.  

    **hist_kwargs : optional
        Additional kwargs that will be used in the matplotlib pyplot hist function.
    """
    if ignore_outliers:
        df = remove_outliers(df, target_col)

    fig, ax_bottom, ax_top = get_dual_gridspec(
        width=width, 
        height=height, 
        margin_plot_height=48, 
        margin_plot_spacer=4)
    
    if hue_order is None:
        hue_order = list(df[hue].unique())
    
    class_top = hue_order[0]
    class_bottom = hue_order[-1]
    
    if (hue==None) or (len(hue_order) != 2):
        raise ValueError('hist_mirrored requires that df[hue] contains exactly two classes')
    
    df_top = df[df[hue]==class_top]
    df_bottom = df[df[hue]==class_bottom]
        
    # If matplotlib arguments are not given, use following defaults
    if 'density' not in hist_kwargs.keys():
        hist_kwargs['density'] = True
    if 'bins' not in hist_kwargs.keys():
        hist_kwargs['bins'] = 100

    ax_top.hist(df_top[target_col], color=color_top, **hist_kwargs)
    ax_bottom.hist(df_bottom[target_col], color=color_bottom, **hist_kwargs)
    ax_bottom.invert_yaxis()
    
    ax_top.set_ylabel(class_top)
    ax_bottom.set_ylabel(class_bottom)
    
    ax_bottom.set_xlabel(target_col)
    
    return fig, ax_top, ax_bottom


def hist_overlay(
    df,
    target_col,
    hue=None,
    width=6,
    height=4,
    hue_order=None,
    group_colors=None,
    cmap='tab10',
    ignore_outliers=True,
    **hist_kwargs
):
    """
    This function plots a an overlaid histogram for data, split on the hue column.  
    
    Parameters
    ----------
    df : pd.DataFrame

    target_col : str 
        The column name that contains the values to be plotted.

    hue : str
        The column name that contains the binary split values.  This column must
        contain exactly two unique values.

    width : int,
        Width of the figure in inches. 

    height : int
        Height of the figure in inches

    hue_order : list
        If given, the top plot will contain the histogram where the hue value equals
        the first item in hue_order, the bottom plot will contain the histogram 
        where the hue value equals the last item in hue_order.

    ignore_outliers : boolean
        If True, the histogram will only consider values between 1st percentile
        and 99th percentile.  

    **hist_kwargs : optional
        Additional kwargs that will be used in the matplotlib pyplot hist function.
    """
    if ignore_outliers:
        df = remove_outliers(df, target_col)

    fig, ax = plt.subplots(figsize=(width, height))
    
    if hue_order is None:
        hue_order = list(df[hue].unique())
    
    if (hue==None):
        raise ValueError('Must provide the hue column')
    
    # Get group colors
    if group_colors==None:
        group_colors = get_plot_colors(hue_order, cmap)

    # If alpha is not provided, set default depending on histtype
    if 'alpha' not in hist_kwargs.keys():
        if 'histtype' in hist_kwargs.keys() and hist_kwargs['histtype'] == 'step':
            hist_kwargs['alpha'] = 1
        else:
            hist_kwargs['alpha'] = 0.6

    # If matplotlib arguments are not given, use following defaults
    if 'density' not in hist_kwargs.keys():
        hist_kwargs['density'] = True
    if 'bins' not in hist_kwargs.keys():
        hist_kwargs['bins'] = 100
        
    for current_hue in hue_order:
        df_current_hue = df[df[hue]==current_hue]
        ax.hist(df_current_hue[target_col], color=group_colors[current_hue], 
                label=current_hue, **hist_kwargs)
    
    leg = get_legend(ax)
    ax.set_xlabel(target_col)

    return fig, ax


def plot_histograms_by_class(
    df,
    dimension_cols,
    hue,
    output_dir=None,
    width=6,
    height=4,
    hue_order=None,
    group_colors=None,
    hist_style='overlay',
    bar_kwargs={},
    hist_kwargs={}
):
    """
    This function plots a histogram for each given column in the dataframe,
    segmented by the hue column.
    
    Parameters
    ----------
    df : pd.DataFrame

    dimension_cols : list 
        List of column names.  A histogram will be plotted for each column name.

    hue : str
        The column name that contains a dimension for splitting the histogram.

    
    width : str
        Path to the folder for saving output figures

    width : int,
        Width of the figure in inches. 

    height : int
        Height of the figure in inches

    hue_order : list
        If given, the top plot will contain the histogram where the hue value equals
        the first item in hue_order, the bottom plot will contain the histogram 
        where the hue value equals the last item in hue_order.

    ignore_outliers : boolean
        If True, the histogram will only consider values between 1st percentile
        and 99th percentile.  

    **hist_kwargs : optional
        Ad
    """
    # If output directory does not exist, create the directory
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)
    
    figures = []
    for dim in dimension_cols:
        
        try:
            if df[dim].dtype in ['str', 'O']:
                dfg = df.groupby([dim, hue]).size().reset_index(name='N')
                fig, group_colors = shares_plot(
                    data=dfg,
                    x=dim,
                    y='N',
                    hue=hue,
                    xtick_rotation=0,
                    normalize=True,
                    cmap='tab10',
                    kind='bar',
                    hue_order=hue_order,
                    **bar_kwargs)

                if output_dir:
                    fig.savefig(f"{output_dir}/cat_{dim}.png", bbox_inches='tight')
                    plt.close()
                else:
                    figures.append(fig)
            elif hist_style=='mirror':
                fig, ax_bottom, ax_top = hist_mirrored(
                    df,
                    target_col=dim,
                    hue=hue,
                    hue_order=hue_order,
                    **hist_kwargs)
                if output_dir:
                    fig.savefig(f"{output_dir}/hist_mirror_{dim}.png", bbox_inches='tight')
                    plt.close()
                else:
                    figures.append(fig)
            elif hist_style=='overlay':
                fig, ax = hist_overlay(
                    df,
                    target_col=dim,
                    hue=hue,
                    group_colors=group_colors,
                    hue_order=hue_order,
                    **hist_kwargs)
                if output_dir:
                    fig.savefig(f"{output_dir}/hist_{dim}.png", bbox_inches='tight')
                    plt.close()
                else:
                    figures.append(fig)
            elif hist_style=='all':
                # Overlay 
                fig, ax = hist_overlay(
                    df,
                    target_col=dim,
                    hue=hue,
                    group_colors=group_colors,
                    hue_order=hue_order,
                    **hist_kwargs)
                if output_dir:
                    fig.savefig(f"{output_dir}/hist_den_{dim}.png", bbox_inches='tight')
                    plt.close()
                else:
                    figures.append(fig)

                # Overlay cumulative
                fig, ax = hist_overlay(
                    df,
                    target_col=dim,
                    hue=hue,
                    group_colors=group_colors,
                    cumulative=True,
                    histtype='step',
                    hue_order=hue_order,
                    **hist_kwargs)
                if output_dir:
                    fig.savefig(f"{output_dir}/hist_cum_{dim}.png", bbox_inches='tight')
                    plt.close()
                else:
                    figures.append(fig)

                # Mirror if binary
                if len(list(df[hue].unique()))==2:
                    fig, ax_bottom, ax_top = hist_mirrored(
                        df,
                        target_col=dim,
                        hue=hue,
                        hue_order=hue_order,
                        **hist_kwargs)
                    if output_dir:
                        fig.savefig(f"{output_dir}/hist_mirror_{dim}.png", bbox_inches='tight')
                        plt.close()
                    else:
                        figures.append(fig)
            else:
                raise ValueError("hist_style must be in ['mirror', 'overlay', 'all']")

        except KeyError:
            logger.warn(f"Column '{dim}' not found in dataframe")

    return figures


