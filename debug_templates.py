import sys
import time
from matplotlib import pyplot as plt
import os

from src.data_loaders import Map, set_settings
from src.create_traj import CreateTraj
from src.noise_traj import NoiseTraj
from src.estimators import IEKF, UKF
from src.outputs_utils import Errors, Covariances, plot_results

"""
    DEBUG
"""
time_start = time.time()
args = set_settings()
map_data = Map(args).load()
print("--- %s seconds ---" % (time.time() - time_start))
print("0")
args.psi = 45
t_traj_0 = CreateTraj(args).linear(map_data)
m_traj_0 = NoiseTraj(t_traj_0).noise(args.imu_errors, dist=args.noise_type)
estimation_results_0 = IEKF(args).run(map_data, m_traj_0)
e_traj_0 = estimation_results_0.traj
print("1")
args.psi = 22
t_traj_1 = CreateTraj(args).linear(map_data)
m_traj_1 = NoiseTraj(t_traj_1).noise(args.imu_errors, dist=args.noise_type)
estimation_results_1 = IEKF(args).run(map_data, m_traj_1)
e_traj_1 = estimation_results_1.traj
print("2")
args.psi = 0
t_traj_2 = CreateTraj(args).linear(map_data)
m_traj_2 = NoiseTraj(t_traj_2).noise(args.imu_errors, dist=args.noise_type)
estimation_results_2 = IEKF(args).run(map_data, m_traj_2)
e_traj_2 = estimation_results_2.traj

# plot for debug
time_end = time.time()
time_elapsed = time_end - time_start
print('Time elapsed: ', time_elapsed)
t_vec = t_traj_0.time_vec

t_lon_0, t_lat_0, t_north_0, t_east_0 = t_traj_0.pos.lon, t_traj_0.pos.lat, t_traj_0.pos.north, t_traj_0.pos.east
t_lon_1, t_lat_1, t_north_1, t_east_1 = t_traj_1.pos.lon, t_traj_1.pos.lat, t_traj_1.pos.north, t_traj_1.pos.east
t_lon_2, t_lat_2, t_north_2, t_east_2 = t_traj_2.pos.lon, t_traj_2.pos.lat, t_traj_2.pos.north, t_traj_2.pos.east

m_lon_0, m_lat_0, m_north_0, m_east_0 = m_traj_0.pos.lon, m_traj_0.pos.lat, m_traj_0.pos.north, m_traj_0.pos.east
m_lon_1, m_lat_1, m_north_1, m_east_1 = m_traj_1.pos.lon, m_traj_1.pos.lat, m_traj_1.pos.north, m_traj_1.pos.east
m_lon_2, m_lat_2, m_north_2, m_east_2 = m_traj_2.pos.lon, m_traj_2.pos.lat, m_traj_1.pos.north, m_traj_1.pos.east

e_lon_0, e_lat_0, e_north_0, e_east_0 = e_traj_0.pos.lon, e_traj_0.pos.lat, e_traj_0.pos.north, e_traj_0.pos.east
e_lon_1, e_lat_1, e_north_1, e_east_1 = e_traj_1.pos.lon, e_traj_1.pos.lat, e_traj_1.pos.north, e_traj_1.pos.east
e_lon_2, e_lat_2, e_north_2, e_east_2 = e_traj_2.pos.lon, e_traj_2.pos.lat, e_traj_2.pos.north, e_traj_2.pos.east

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 12))

# Subplot 1: Latitude Comparison
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(t_vec, e_lat_0-e_lat_2, label='e_lat_0', color='blue')
plt.plot(t_vec, e_lat_1-e_lat_2, label='e_lat_1', color='red')
plt.plot(t_vec, e_lat_2-e_lat_1, label='e_lat_2', color='green')
plt.title('Comparison of Estimated Latitudes')
plt.xlabel('Time')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)

# Subplot 2: Longitude Comparison
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(t_vec, e_lon_0-e_lon_1, label='e_lon_0', color='blue')
plt.plot(t_vec, e_lon_1-e_lon_2, label='e_lon_1', color='red')
plt.plot(t_vec, e_lon_2-e_lon_1, label='e_lon_2', color='green')
plt.title('Comparison of Estimated Longitudes')
plt.xlabel('Time')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
plt.savefig('DEBUG_Lat_Lon.png')
plt.show()

plt.figure(figsize=(20, 12))

# Subplot 1: Latitude Comparison
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(t_vec, e_north_0-e_north_2, label='e_north_0', color='blue')
plt.plot(t_vec, e_north_1-e_north_2, label='e_north_1', color='red')
plt.plot(t_vec, e_north_2-e_north_0, label='e_north_2', color='green')
plt.title('Comparison of Estimated North')
plt.xlabel('Time')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)

# Subplot 2: Longitude Comparison
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(t_vec, e_east_0-e_east_1, label='e_east_0', color='blue')
plt.plot(t_vec, e_east_1-e_east_2, label='e_east_1', color='red')
plt.plot(t_vec, e_east_2-e_east_0, label='e_east_2', color='green')
plt.title('Comparison of Estimated East')
plt.xlabel('Time')
plt.ylabel('East')
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
plt.savefig('DEBUG_North_East.png')
plt.show()

pass
"""
    DEBUG
"""