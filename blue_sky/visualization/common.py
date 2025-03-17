import matplotlib.pyplot as plt
import os
import numpy as np


def setup_figure(title, figsize=(10, 8)):
    """
    Create a figure with the given title and size.

    Args:
        title (str): Title of the figure
        figsize (tuple): Figure size as (width, height)

    Returns:
        tuple: (fig, title) where fig is the matplotlib figure and title is the figure title
    """
    fig = plt.figure(title, figsize=figsize)
    fig.suptitle(title, fontsize=24, fontweight='bold')
    return fig, title


def save_figure(fig, title, output_folder, formats=('png', 'svg')):
    """
    Save a figure to disk in multiple formats.

    Args:
        fig: Matplotlib figure
        title (str): Figure title (used for filename)
        output_folder (str): Folder to save the figure in
        formats (tuple): File formats to save as
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the figure in each format
    for fmt in formats:
        filename = os.path.join(output_folder, f'{title}.{fmt}')
        fig.savefig(filename)

    return fig


def create_error_plot(time_vec, errors, covariances, title, ylabel, subplot=None):
    """
    Create a standard error plot with error and covariance bounds.

    Args:
        time_vec (numpy.ndarray): Time vector for x-axis
        errors (numpy.ndarray): Error values to plot
        covariances (numpy.ndarray): Covariance values for bounds
        title (str): Title for the subplot
        ylabel (str): Y-axis label
        subplot: Matplotlib axis to plot on (optional)

    Returns:
        matplotlib.axes.Axes: The subplot with the plot
    """
    if subplot is None:
        _, ax = plt.subplots(1, 1)
    else:
        ax = subplot

    # Plot error and confidence bounds
    ax.plot(time_vec, errors, '-r', linewidth=1)
    ax.plot(time_vec, covariances, '--b', linewidth=1)
    ax.plot(time_vec, -covariances, '--b', linewidth=1)

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(['Error', r'$\pm\sigma$'], loc='upper right')

    return ax


def compute_plot_limits(data_sequences, margin_percent=5):
    """
    Compute plot limits with margins for multiple data sequences.

    Args:
        data_sequences (list): List of data arrays
        margin_percent (float): Percentage margin to add

    Returns:
        tuple: (min_value, max_value) with margins
    """
    min_val = np.min([np.min(seq) for seq in data_sequences])
    max_val = np.max([np.max(seq) for seq in data_sequences])

    margin = (max_val - min_val) * margin_percent / 100
    return min_val - margin, max_val + margin