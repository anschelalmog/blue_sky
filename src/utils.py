from numpy import sin, cos, pi, power, array, radians


def set_errors(args, imu_errors):
    args.initial_position_error = args.val_err_pos * imu_errors['initial_position']
    args.velocity_error = args.val_err_vel * imu_errors['velocity']
    args.euler_angles_error = args.val_err_eul * imu_errors['euler_angles']
    args.barometer_bias_error = args.val_err_baro_bias * imu_errors['barometer_bias']
    args.barometer_noise_error = args.val_err_baro_noise * imu_errors['barometer_noise']
    args.altimeter_error = args.val_err_alt * imu_errors['altimeter_noise']


def get_mpd(lat, h=0):
    r0 = 6378137
    f = 0.00335281066474
    lat = array(radians(lat))

    rn = r0 * (1.0 - f * (2.0 - 3.0 * power(sin(lat), 2))) + h  # [m]
    re = r0 * (1.0 + f * power(sin(lat), 2)) + h  # [m]

    mpd_n = rn * pi / 180  # [m/deg]
    mpd_e = re * cos(lat) * pi / 180  # [m/deg]

    return mpd_n, mpd_e  # [m/deg]


def cosd(angel):
    return cos(radians(angel))


def sind(angel):
    return sin(radians(angel))


class DCM:
    """
    This class represents a Direction Cosine Matrix (DCM) that is used to perform rotations in three dimensions.
    It converts Euler angles (yaw, pitch, and roll) into a DCM and allows for the application of this matrix to
    specific rotation axes ("north", "east", "down").

    The Euler angles are interpreted according to the aerospace convention (yaw around the z-axis, pitch around the
    y-axis, and roll around the x-axis). The calculations are based on the formalisms provided on Wikipedia's page on
    rotation formalisms in three dimensions.

    Attributes: yaw (float): The yaw angle in degrees, representing rotation around the z-axis. pitch (float): The
    pitch angle in degrees, representing rotation around the y-axis. roll (float): The roll angle in degrees,
    representing rotation around the x-axis. rotation_axis (str, optional): The axis around which the final rotation
    is applied. Accepted values are "north", "east", or "down". _dcm (numpy.ndarray): The computed Direction Cosine
    Matrix based on the provided Euler angles.

    Methods: _calculate_dcm(): Computes the DCM based on the initialized Euler angles. rot_axis(rotation_axis):
    Applies the DCM to a specified axis, returning a vector representing the rotation around that axis. matrix: A
    property that returns the computed DCM.

    taken from 'https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions'
    Usage:
        # To create a DCM object and get the DCM matrix
        dcm = DCM(yaw=30, pitch=45, roll=60)
        dcm_matrix = dcm. matrix

        # To apply the DCM to a specific axis
        rotated_vector = dcm.rot_axis("north")
    """
    def __init__(self, yaw: float, pitch: float, roll: float, rotation_axis: str = None):
        self.yaw = radians(yaw)
        self.pitch = radians(pitch)
        self.roll = radians(roll)
        self.rotation_axis = rotation_axis
        self._dcm = self._calculate_dcm()

    def _calculate_dcm(self):
        # Yaw (around z-axis)
        rot_z = array([[cos(self.yaw), -sin(self.yaw), 0],
                       [sin(self.yaw), cos(self.yaw), 0],
                       [0, 0, 1]])

        # Pitch (around y-axis)
        rot_y = array([[cos(self.pitch), 0, sin(self.pitch)],
                       [0, 1, 0],
                       [-sin(self.pitch), 0, cos(self.pitch)]])

        # Roll (around x-axis)
        rot_x = array([[1, 0, 0],
                       [0, cos(self.roll), -sin(self.roll)],
                       [0, sin(self.roll), cos(self.roll)]])

        return rot_z @ rot_y @ rot_x

    def _rot_axis(self, rotation_axis: str):
        assert rotation_axis in ["north", "east", "down"], "Invalid rotation axis"
        axis_vector = [rotation_axis == "north", rotation_axis == "east", rotation_axis == "down"]
        return self._dcm @ axis_vector

    def rot_north(self):
        return self._rot_axis("north")[0]

    def rot_east(self):
        return self._rot_axis("east")[1]

    def rot_down(self):
        return self._rot_axis("down")[2]

    @property
    def matrix(self):
        return self._dcm
