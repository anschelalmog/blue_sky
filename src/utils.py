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
    re = r0 * (1.0 + f * power(sin(lat), 2)) + h   # [m]

    mpd_n = rn * pi / 180  # [m/deg]
    mpd_e = re * cos(lat) * pi / 180  # [m/deg]

    return mpd_n, mpd_e  # [m/deg]


def cosd(angel):
    return cos(radians(angel))


def sind(angel):
    return sin(radians(angel))
