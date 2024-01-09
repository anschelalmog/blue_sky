from numpy import zeros
class BasePos:
    def __init__(self, length):
        self.lat = zeros(length)
        self.lon = zeros(length)
        self.north = zeros(length)
        self.east = zeros(length)
        self.h_agl = zeros(length)
        self.h_asl = zeros(length)
        self.h_map = zeros(length)


class BaseVel:
    def __init__(self, length):
        self.north = zeros(length)
        self.east = zeros(length)
        self.down = zeros(length)


class BaseEuler:
    def __init__(self, length):
        self.psi = zeros(length)
        self.theta = zeros(length)
        self.phi = zeros(length)


class BaseTraj:
    def __init__(self, length):
        self.pos = BasePos(length)
        self.vel = BaseVel(length)
        self.euler = BaseEuler(length)
