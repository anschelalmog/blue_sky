"""
Factory module for creating different Kalman filter implementations.
This allows for easy switching between implementations without changing the main code.
"""

def create_filter(filter_type, args):
    """
    Create a Kalman filter instance based on the specified type.

    Args:
        filter_type (str): The type of filter to create - 'IEKF', 'UKF', 'FilterPyEKF', 'FilterPyUKF'
        args: Configuration arguments

    Returns:
        Filter instance based on the specified type

    Raises:
        ValueError: If an invalid filter type is specified
    """
    if filter_type == 'IEKF':
        from .iekf import IEKF
        return IEKF(args)
    elif filter_type == 'UKF':
        from .ukf import UKF
        return UKF(args)
    elif filter_type == 'FilterPyEKF':
        from .iekf_filterpy import FilterPyEKF
        return FilterPyEKF(args)
    elif filter_type == 'FilterPyUKF':
        from .ukf_filterpy import FilterPyUKF
        return FilterPyUKF(args)
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")