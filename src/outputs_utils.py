import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from src.base_traj import BaseTraj


# noinspection DuplicatedCode


class Errors(BaseTraj):
    def __init__(self, used_traj, est_traj):
        super().__init__(used_traj.run_points)
        self.pos.north = used_traj.pos.north - est_traj.pos.lat * used_traj.mpd_north
        self.pos.east = used_traj.pos.east - est_traj.pos.lon * used_traj.mpd_east
        self.pos.h_asl = used_traj.pos.h_asl - est_traj.pos.h_asl
        #
        self.vel.north = used_traj.vel.north - est_traj.vel.north
        self.vel.east = used_traj.vel.east - est_traj.vel.east
        self.vel.down = used_traj.vel.down - est_traj.vel.down
        #
        # self.euler.psi = used_traj.euler.psi - est_traj.euler.psi
        # self.euler.theta = used_traj.euler.theta - est_traj.euler.theta
        # self.euler.phi = used_traj.euler.phi - est_traj.euler.phi


class Covariances(BaseTraj):
    def __init__(self, covariances):
        super().__init__(covariances.shape[2])
        self.pos.north = np.sqrt(covariances[0, 0, :])
        self.pos.east = np.sqrt(covariances[1, 1, :])
        self.pos.h_asl = np.sqrt(covariances[2, 2, :])
        #
        self.vel.north = np.sqrt(covariances[3, 3, :])
        self.vel.east = np.sqrt(covariances[4, 4, :])
        self.vel.down = np.sqrt(covariances[5, 5, :])
        #
        # self.euler.psi = P_est[6, 6, :]
        # self.euler.theta = P_est[7, 7, :]
        # self.euler.phi = P_est[8, 8, :]


def plot_results(args, map_data, ground_truth, measurements, estimation_results, errors, covars):
    os.makedirs('Results', exist_ok=True)
    repr(args.plots)

    if args.plots['plot map']:
        fig = plt.figure('results_on_map', figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('MAP', fontsize=24, fontweight='bold')
        X, Y = np.meshgrid(map_data.ax_lon, map_data.ax_lat)
        ax.plot_surface(X, Y, map_data.grid, cmap='bone')
        plt.grid(False)
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_zlabel('Height [m]')

        # Ground Truth
        if ground_truth is not None:
            ax.plot3D(ground_truth.pos.lon, ground_truth.pos.lat, ground_truth.pos.h_asl, linewidth=4, color='r',
                      label='Ground Truth')

        # Measured
        idx = 10  # every 10th element
        ax.scatter3D(measurements.pos.lon[::idx], measurements.pos.lat[::idx], measurements.pos.h_asl[::idx],
                     marker='x', color='black', label='Measured')

        # Estimated
        ax.plot3D(estimation_results.traj.pos.lon, estimation_results.traj.pos.lat, estimation_results.traj.pos.h_asl,
                  linewidth=4, color='b', label='Estimated')

        # Legend
        lgd = ax.legend(loc='best')
        lgd.set_title('PATHS')
        lgd.get_frame().set_linewidth(1.0)
        plt.tight_layout()

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'map_plot.png'))
        plt.savefig(os.path.join(args.results_folder, 'map_plot.svg'))

        # show fig
        plt.show()
    if args.plots['position errors']:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Position Errors', fontsize=24, fontweight='bold')

        # Error in North position
        axs[0].plot(args.time_vec, errors.pos.north, '-r', linewidth=1)
        axs[0].plot(args.time_vec, covars.pos.north, '--b', linewidth=1)
        axs[0].plot(args.time_vec, -covars.pos.north, '--b', linewidth=1)
        axs[0].set_title('North Position Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Err', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in East Position
        axs[1].plot(args.time_vec, errors.pos.east, '-r', linewidth=1)
        axs[1].plot(args.time_vec, covars.pos.east, '--b', linewidth=1)
        axs[1].plot(args.time_vec, -covars.pos.east, '--b', linewidth=1)
        axs[1].set_title('North Position Error [m]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Err', r'+$\sigma$', r'-$\sigma$'], loc='best')
        # save fig
        plt.savefig(os.path.join(args.results_folder, 'position_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'position_errors.svg'))

        # plot show
        plt.show()
    if args.plots['velocity errors']:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Velocity Errors', fontsize=24, fontweight='bold')

        # Error in North Velocity
        axs[0].plot(args.time_vec, errors.vel.north, '-r', linewidth=1)
        axs[0].plot(args.time_vec, covars.vel.north, '--b', linewidth=1)
        axs[0].plot(args.time_vec, -covars.vel.north, '--b', linewidth=1)
        axs[0].set_title('North Velocity Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in East Velocity
        axs[1].plot(args.time_vec, errors.vel.east, '-r', linewidth=1)
        axs[1].plot(args.time_vec, covars.vel.east, '--b', linewidth=1)
        axs[1].plot(args.time_vec, -covars.vel.east, '--b', linewidth=1)
        axs[1].set_title('East Position Error [m]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in Down Velocity
        axs[2].plot(args.time_vec, errors.vel.down, '-r', linewidth=1)
        axs[2].plot(args.time_vec, covars.vel.down, '--b', linewidth=1)
        axs[2].plot(args.time_vec, -covars.vel.down, '--b', linewidth=1)
        axs[2].set_title('East Position Error [m]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].grid(True)
        axs[2].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        plt.tight_layout()

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'velocity_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'velocity_errors.svg'))

        plt.show()
    if args.plots['altitude errors']:
        plt.figure('Altitude Errors', figsize=(10, 12))
        plt.suptitle('Altitude Errors', fontsize=24, fontweight='bold')

        plt.plot(args.time_vec, errors.pos.h_asl, 'r', linewidth=1)
        plt.plot(args.time_vec, covars.pos.h_asl, '--b', linewidth=1)
        plt.plot(args.time_vec, -covars.pos.h_asl, '--b', linewidth=1)

        mean_err_alt = np.mean(errors.pos.h_asl)
        plt.plot(args.time_vec, mean_err_alt * np.ones(len(errors.pos.h_asl)), '*-k')
        plt.plot(args.time_vec, estimation_results.params.Z, '-.g')

        # legend
        plt.title('Altitude Err [m]')
        plt.xlabel('Time [sec]')
        plt.grid(True)
        plt.legend(['Error', r'+$\sigma$', r'-$\sigma$', 'mean', 'Z'], loc='best')

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'altitude_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'altitude_errors.svg'))

        # show plot
        plt.show()
    if args.plots['attitude errors']:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Attitude Errors', fontsize=24, fontweight='bold')

        # Error in euler psi
        axs[0].plot(args.time_vec, errors.euler.psi, '-r', linewidth=1)
        axs[0].plot(args.time_vec, covars.euler.psi, '--b', linewidth=1)
        axs[0].plot(args.time_vec, -covars.euler.psi, '--b', linewidth=1)
        axs[0].set_title('Euler Psi Error [deg]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in euler theta
        axs[1].plot(args.time_vec, errors.euler.theta, '-r', linewidth=1)
        axs[1].plot(args.time_vec, covars.euler.theta, '--b', linewidth=1)
        axs[1].plot(args.time_vec, -covars.euler.theta, '--b', linewidth=1)
        axs[1].set_title('Euler Theta Error [deg]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in euler phi
        axs[2].plot(args.time_vec, errors.euler.phi, '-r', linewidth=1)
        axs[2].plot(args.time_vec, covars.euler.phi, '--b', linewidth=1)
        axs[2].plot(args.time_vec, -covars.euler.phi, '--b', linewidth=1)
        axs[2].set_title('East Position Error [m]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].grid(True)
        axs[2].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        plt.tight_layout()

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'attitude_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'attitude_errors.svg'))

        # plot show
        plt.show()
    if args.plots['model errors']:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Model Errors', fontsize=24, fontweight='bold')

        axs[0].set_title('Measurement mismatch and Error correction')
        axs[0].plot(args.time_vec, estimation_results.params.Z, '--b')
        axs[0].plot(args.time_vec, estimation_results.params.dX[2, :], '-r')
        axs[0].set_ylabel('Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].set_ylim(-1, 1.1)
        axs[0].legend(['Error', 'Mismatch'], loc='best')

        axs[1].set_title('Process Noise, R and Rc')
        axs[1].plot(args.time_vec, estimation_results.params.Rc, '-r')
        axs[1].plot(args.time_vec, estimation_results.params.Rfit, '--b')
        axs[0].set_ylabel('Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Rc - penalty on height', 'Rfit', 'R'], loc='best')

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'model_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'model_errors.svg'))

        # show plot
        plt.show()
    if args.plots['kalman gains']:
        gains = [
            'Position North',
            'Position East',
            'Position Down',
            'Velocity North',
            'Velocity East',
            'Velocity Down'
        ]

        fig, axs = plt.subplots(2, 3, figsize=(10, 12))
        fig.suptitle('Kalman Gains', fontsize=24, fontweight='bold')

        for i, title in enumerate(gains):
            row = i // 3
            col = i % 3
            axs[row, col].set_ylim(-1, 1.1)
            axs[row, col].grid(True)
            axs[row, col].plot(args.time_vec, estimation_results.params.K[i, :], linewidth=1)
            axs[row, col].set_title(title)
            axs[row, col].set_xlabel('Time [sec]')

        plt.tight_layout()
        # save fig
        plt.savefig(os.path.join(args.results_folder, 'kalman_gains.png'))
        plt.savefig(os.path.join(args.results_folder, 'kalman_gains.svg'))

        # plot show
        plt.show()
    if args.plots['map elevation']:
        fig = plt.figure('Map  - Ground Elevation at PinPoint', figsize=(10, 12))
        fig.suptitle('Ground Elevation at PinPoint', fontsize=24, fontweight='bold')
        plt.plot(args.time_vec, ground_truth.pinpoint.h_map, 'r')
        plt.plot(args.time_vec, estimation_results.traj.pos.h_asl - estimation_results.traj.pos.h_agl, '--b')
        plt.title('Map elevation')
        plt.title('Height [m]')
        plt.xlabel('Time [sec]')
        plt.grid(True)
        plt.legend(['True', 'Estimated'])

        plt.tight_layout()
        # save fig
        plt.savefig(os.path.join(args.results_folder, 'map_elevation.png'))
        plt.savefig(os.path.join(args.results_folder, 'map_elevation.svg'))

        # show plot
        plt.show()

    return errors, covars



def print_log(args, estimation_results, errors, covs):
    # Create a log file in the Results folder
    results_folder = args.results_folder
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%d%m%Y_%H%M')
    log_file_name = f'log_{formatted_time}.txt'
    log_file_path = os.path.join(results_folder, log_file_name)

    with open(log_file_path, 'w') as log_file:
        # Write header information
        log_file.write("========================================\n")
        log_file.write("             Trajectory Run Log         \n")
        log_file.write("========================================\n")
        log_file.write(f"Date and Time of Run: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
        log_file.write("----------------------------------------\n")
        # log_file.write("Run Configuration:\n")
        # log_file.write(f"{args}\n")
        # log_file.write("----------------------------------------\n\n")

        if args.noise_type == 'normal' or args.noise_type == 'uniform':
            log_file.write(f"noising the trajectory with {args.noise_type} noise\n")
        else:  # no noise
            log_file.write("trajectory with no noise\n")


        # est_traj = errors
        # covariance = covs
        #
        # for key in ['north', 'east', 'down']:
        #     pos_error_within_cov = np.mean(np.abs(est_traj.pos[key]) <= np.sqrt(covariance.pos[key])) * 100
        #     vel_error_within_cov = np.mean(np.abs(est_traj.vel[key]) <= np.sqrt(covariance.vel[key])) * 100
        #     log_file.write(
        #         f"Percentage of Position {key.capitalize()} Error within Covariance Limit: {pos_error_within_cov:.2f}%\n")
        #     log_file.write(
        #         f"Percentage of Velocity {key.capitalize()} Error within Covariance Limit: {vel_error_within_cov:.2f}%\n")
        #
        # # Additional statistics - RMSE for position and velocity
        # rmse_pos = np.sqrt(np.mean(np.square([est_traj.pos[key] for key in ['north', 'east', 'down']])))
        # rmse_vel = np.sqrt(np.mean(np.square([est_traj.vel[key] for key in ['north', 'east', 'down']])))
        # log_file.write(f"RMSE of Position: {rmse_pos:.2f}\n")
        # log_file.write(f"RMSE of Velocity: {rmse_vel:.2f}\n")
