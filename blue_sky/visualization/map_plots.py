import matplotlib.pyplot as plt
import numpy as np
from .common import setup_figure, save_figure


def plot_map(map_data, mode='2D', save=False, output_folder='out'):
    """
    Visualize the map in either 2D or 3D mode.

    Args:
        map_data: Map data containing grid and axis information
        mode (str): Visualization mode, either '2D' or '3D'
        save (bool): Whether to save the figure
        output_folder (str): Folder to save the figure in

    Returns:
        matplotlib.figure.Figure: The figure
    """
    if map_data.grid is None:
        print("Map grid is not initialized.")
        return None

    if mode == '2D':
        title = 'Map Visualization (2D)'
        fig, title = setup_figure(title, figsize=(10, 10))

        plt.imshow(map_data.grid,
                   extent=(map_data.axis['east'].min(), map_data.axis['east'].max(),
                           map_data.axis['north'].min(), map_data.axis['north'].max()),
                   origin='lower',
                   cmap='terrain')

        plt.title(title)
        plt.xlabel('East')
        plt.ylabel('North')
        plt.colorbar(label='Elevation')
        plt.grid(True)

    elif mode == '3D':
        title = 'Map Visualization (3D)'
        fig, title = setup_figure(title, figsize=(12, 8))

        North, East = np.meshgrid(map_data.axis['north'], map_data.axis['east'])

        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(East, North, map_data.grid.T, cmap='terrain', edgecolor='none')

        ax.set_title(title)
        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('Elevation')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    else:
        print("Invalid mode selected. Please choose '2D' or '3D'.")
        return None

    if save:
        save_figure(fig, title, output_folder)

    return fig