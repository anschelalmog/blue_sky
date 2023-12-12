import numpy as np


def get_mpd(lat, h=0):
    r0 = 6378137
    f = 0.00335281066474
    lat = np.array(np.radians(lat))

    rn = r0 * (1.0 - f * (2.0 - 3.0 * np.power(np.sin(lat), 2))) + h  # [m]
    re = r0 * (1.0 + f * np.power(np.sin(lat), 2)) + h  # [m]

    mpd_n = rn * np.pi / 180  # [m/deg]
    mpd_e = re * np.cos(lat) * np.pi / 180  # [m/deg]

    return mpd_n, mpd_e


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))
