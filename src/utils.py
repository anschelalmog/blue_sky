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


def euler_to_dcm(rotation_axis: str, yaw: float, pitch: float, roll: float):
    """
    Convert Euler angles to Direction Cosine Matrix (DCM).

    taken from 'https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions'
    :param rotation_axis: The axis around which the rotation is performed. Only accepts "north", "east", or "down".
    :param yaw: The yaw angle in degrees.
    :param pitch: The pitch angle in degrees.
    :param roll: The roll angle in degrees.
    :return: The DCM matrix multiplied by the axis vector.
    """
    assert rotation_axis in ["north", "east", "down"]
    axis_vector = [rotation_axis == "north",
                   rotation_axis == "east",
                   rotation_axis == "down"]

    yaw, pitch, roll = radians(yaw), radians(pitch), radians(roll)

    # Yaw (around z-axis)
    rot_z = array([[cos(yaw), -sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [0, 0, 1]])

    # Pitch (around y-axis)
    rot_y = array([[cos(pitch), 0, sin(pitch)],
                   [0, 1, 0],
                   [-sin(pitch), 0, cos(pitch)]])

    # Roll (around x-axis)
    rot_x = array([[1, 0, 0],
                   [0, cos(roll), -sin(roll)],
                   [0, sin(roll), cos(roll)]])

    # Combined DCM
    dcm = rot_z @ rot_y @ rot_x
    return dcm @ axis_vector
