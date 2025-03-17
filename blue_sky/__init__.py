# Import all essential components for backward compatibility
from blue_sky.config.settings import set_settings
from blue_sky.core.base import BaseTraj, BasePos, BaseVel, BaseEuler
from blue_sky.core.map import Map
from blue_sky.core.trajectory import CreateTraj
from blue_sky.core.pinpoint import PinPoint
from blue_sky.estimation.iekf import IEKF
from blue_sky.estimation.ukf import UKF
from blue_sky.simulation.noise import NoiseTraj
from blue_sky.simulation.errors import IMUErrors, Covariances, RunErrors, calc_errors_covariances
from blue_sky.visualization.combined import plot_results
from blue_sky.visualization.trajectory_plots import plot_trajectory_comparison, plot_pinpoint_trajectories_2d, plot_pinpoint_trajectories_3d

# Export all components
__all__ = [
    'set_settings',
    'BaseTraj', 'BasePos', 'BaseVel', 'BaseEuler',
    'Map', 'CreateTraj', 'PinPoint',
    'IEKF', 'UKF',
    'NoiseTraj', 'IMUErrors', 'Covariances', 'RunErrors', 'calc_errors_covariances',
    'plot_results', 'plot_trajectory_comparison', 'plot_pinpoint_trajectories_2d', 'plot_pinpoint_trajectories_3d'
]