from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..core.base import  BaseTraj
from ..core.pinpoint import PinPoint
from ..utils.math import  cosd, sind


class BaseKalmanFilter(ABC):
    """
    Abstract base class for Kalman filter implementations.

    This class provides common functionality for different Kalman filter
    variants used for trajectory estimation.

    Attributes:
        run_points (int): Number of points in the trajectory.
        time_vec (numpy.ndarray): Time vector for the trajectory.
        del_t (float): Time step (resolution).
        state_size (int): Size of the state vector.
        curr_state (int): Current state index during estimation.
        traj (BaseTraj): Trajectory instance to store estimation results.
    """

    def __init__(self, args):
        """
        Initialize the base Kalman filter.

        Args:
            args: Configuration arguments containing filter parameters.
        """
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        self.del_t = args.time_res
        self.state_size = args.kf_state_size
        self.curr_state = None
        self.traj = BaseTraj(self.run_points)
        self.traj.pinpoint = PinPoint(self.run_points)

    @abstractmethod
    def run(self, map_data, meas):
        """
        Run the filter to estimate trajectory.

        Args:
            map_data: Map containing terrain information
            meas: Measurements to use for estimation

        Returns:
            self: Updated instance with estimation results
        """
        pass

    def _initialize_traj(self, meas):
        """
        Set initial values from measurements.

        Args:
            meas: Measurement trajectory with initial values.
        """
        self.traj.pos.east[0] = meas.pos.east[0]
        self.traj.pos.north[0] = meas.pos.north[0]
        self.traj.pos.lat[0] = meas.pos.lat[0]
        self.traj.pos.lon[0] = meas.pos.lon[0]
        self.traj.pos.h_asl[0] = meas.pos.h_asl[0]  # altitude
        jac = cosd(meas.euler.theta[0]) * cosd(meas.euler.phi[0])
        self.traj.pos.h_agl[0] = meas.pinpoint.range[0] * jac  # H_agl_p

        self.traj.vel.north[0] = meas.vel.north[0]
        self.traj.vel.east[0] = meas.vel.east[0]
        self.traj.vel.down[0] = meas.vel.down[0]

        self.traj.pinpoint.delta_north[0] = meas.pinpoint.delta_north[0]
        self.traj.pinpoint.delta_east[0] = meas.pinpoint.delta_east[0]
        self.traj.pinpoint.h_map[0] = meas.pinpoint.h_map[0]

        self.traj.euler.psi[0] = meas.euler.psi[0]
        self.traj.euler.theta[0] = meas.euler.theta[0]
        self.traj.euler.phi[0] = meas.euler.phi[0]

    def _pinpoint_coordinates(self, meas):
        """
        Calculate current lat, lon for pinpoint location.

        Args:
            meas: Measurement trajectory.

        Returns:
            tuple: (latitude, longitude) coordinates.
        """
        i = self.curr_state
        psi = self.traj.euler.psi[i]  # Use estimated orientation
        theta = self.traj.euler.theta[i]
        phi = self.traj.euler.phi[i]

        # Calculate offsets based on orientation and range
        self.traj.pinpoint.delta_north[i] = meas.pinpoint.range[i] * (
                cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi))
        self.traj.pinpoint.delta_east[i] = meas.pinpoint.range[i] * (
                sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi))

        # Calculate lat/lon at pinpoint
        lat = self.traj.pos.lat[i] + self.traj.pinpoint.delta_north[i] / meas.mpd_north[i]
        lon = self.traj.pos.lon[i] + self.traj.pinpoint.delta_east[i] / meas.mpd_east[i]

        # Store calculated coordinates for use in next iteration
        self.traj.pinpoint.lat[i] = lat
        self.traj.pinpoint.lon[i] = lon

        return lat, lon

    def _find_slopes(self, lat, lon, p_pre, map_data):
        """
        Calculate the north-south and east-west slopes of terrain at a given lat lon,
        and estimate the Error of the slope model.

        Args:
            lat: Latitude at point [deg]
            lon: Longitude at point [deg]
            p_pre: A priori Error covariance matrix
            map_data: MapLoad instance that contain the grid and its axes

        Updates:
            SN, SE: Slope values in params
            RFIT: Fit error in params
        """
        i = self.curr_state
        dP = 100  # distance increments in [m]
        delPmap = np.array([dP / map_data.mpd['north'][i], dP / map_data.mpd['east'][i]])  # [deg]

        # max number of points in each direction
        maxP = np.sqrt(max(p_pre[0][0], p_pre[1][1]))
        KP = 3
        NC = int(np.ceil(max(KP, 2 * np.ceil(KP * maxP / dP) + 1) / 2))
        idx = int((NC - 1) / 2)  # indices

        # create lat lon vectors according to grid indices
        pos_offset = np.arange(-idx, idx + 1)
        lat_vec = delPmap[0] * pos_offset + lat
        lon_vec = delPmap[1] * pos_offset + lon

        # Check if lat/lon are within map bounds
        lat_min, lat_max = np.min(map_data.axis['lat']), np.max(map_data.axis['lat'])
        lon_min, lon_max = np.min(map_data.axis['lon']), np.max(map_data.axis['lon'])

        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            # Outside map bounds, use safe defaults
            self.params.SN[i], self.params.SE[i] = 0, 0
            self.params.Rfit[i] = 1000  # High uncertainty when interpolation fails
            return

        # Create the meshgrid
        yp, xp = np.meshgrid(lat_vec, lon_vec)

        # scaling factors for slope calc
        sx2 = (dP ** 2) * 2 * NC * np.sum(np.power(np.arange(1, idx + 1), 2))
        sy2 = sx2

        try:
            # interpolate elevation data for current location
            interpolator = RegularGridInterpolator((map_data.axis['lat'], map_data.axis['lon']), map_data.grid,
                                                   bounds_error=False, fill_value=None)

            ref_elevation = interpolator((lat, lon))

            # Reshape points for vectorized interpolation
            points = np.column_stack((yp.flatten(), xp.flatten()))
            elevations = interpolator(points).reshape(yp.shape)

            # Calculate slopes using proper indexing
            syh = dP * np.sum(np.dot(pos_offset, (elevations - ref_elevation)))
            sxh = dP * np.sum(np.dot((elevations - ref_elevation), pos_offset))

            self.params.SN[i], self.params.SE[i] = syh / sy2, sxh / sx2

            # Calculate the Error over the grid
            MP = (2 * idx + 1) ** 2  # number of points in the mesh grid
            In = 0

            for step_E in np.arange(-idx, idx + 1):
                for step_N in np.arange(-idx, idx + 1):
                    north = int(step_N + idx)
                    east = int(step_E + idx)
                    # Fix: use step_E instead of step_N for the east component
                    height_model = dP * (self.params.SN[i] * step_N + self.params.SE[i] * step_E)
                    In += (height_model - elevations[east][north] + ref_elevation) ** 2

            self.params.Rfit[i] = In / (MP - 1)

        except (ValueError, IndexError) as e:
            print(f"Error in _find_slopes at index {i}: {e}")
            self.params.SN[i], self.params.SE[i] = 0, 0
            self.params.Rfit[i] = 1000  # High uncertainty when interpolation fails

    def _calc_rc(self, h_agl_meas):
        """
        Calculate the measurement noise covariance based on height above ground.

        Args:
            h_agl_meas: Height above ground level
        """
        i = self.curr_state

        if h_agl_meas <= 200:
            self.params.Rc[i] = 100
        elif h_agl_meas <= 760:
            self.params.Rc[i] = 225  # 100 + 125
        elif h_agl_meas <= 1000:
            self.params.Rc[i] = 400  # 225 + 175
        elif h_agl_meas <= 5000:
            self.params.Rc[i] = 1000  # 400 + 600
        elif h_agl_meas <= 7000:
            self.params.Rc[i] = 1500  # 1000 + 500
        else:
            self.params.Rc[i] = 3000  # 1500 + 1500

        # Increase Rc towards the end to account for increasing measurement uncertainty
        if i > self.run_points * 0.8:
            self.params.Rc[i] *= 1.5