import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from src.base_traj import BaseTraj


# noinspection DuplicatedCode


class RunErrors(BaseTraj):
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
        self.euler.psi = used_traj.euler.psi - est_traj.euler.psi
        self.euler.theta = used_traj.euler.theta - est_traj.euler.theta
        self.euler.phi = used_traj.euler.phi - est_traj.euler.phi


class Covariances(BaseTraj):
    def __init__(self, covariances):
        super().__init__(covariances.shape[2])
        #
        self.pos.north = np.sqrt(covariances[0, 0, :])
        self.pos.east = np.sqrt(covariances[1, 1, :])
        self.pos.h_asl = np.sqrt(covariances[2, 2, :])
        #
        self.vel.north = np.sqrt(covariances[3, 3, :])
        self.vel.east = np.sqrt(covariances[4, 4, :])
        self.vel.down = np.sqrt(covariances[5, 5, :])
        #
        self.acc.north = np.sqrt(covariances[6, 6, :])
        self.acc.east = np.sqrt(covariances[7, 7, :])
        self.acc.down = np.sqrt(covariances[8, 8, :])
        #
        self.euler.psi = np.sqrt(covariances[9, 9, :])
        self.euler.theta = np.sqrt(covariances[10, 10, :])
        self.euler.phi = np.sqrt(covariances[11, 11, :])


   # Mapping of attribute names to their corresponding indices in the covariance matrix
        attr_indices = {
            'pos': {'north': 0, 'east': 1, 'h_asl': 2},
            'vel': {'north': 3, 'east': 4, 'down': 5},
            'acc': {'north': 6, 'east': 7, 'down': 8},
            'euler': {'psi': 9, 'theta': 10, 'phi': 11}
        }

        # Iterate over the attribute groups and their items
        for attr_group, indices in attr_indices.items():
            for attr_name, idx in indices.items():
                # Use sqrt to calculate the standard deviation from the variance
                setattr(getattr(self, attr_group), attr_name, np.sqrt(covariances[idx, idx, :]))


def plot_results(args, map_data, ground_truth, measurements, estimation_results, errors, covars):
    os.makedirs('out', exist_ok=True)
    repr(args.plots)

    if args.plots['plot map']:
        title = 'Results on Map'
        fig = plt.figure(title, figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('MAP', fontsize=24, fontweight='bold')
        X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
        ax.plot_surface(X, Y, map_data.grid, cmap='bone')
        plt.grid(False)
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_zlabel('Height [m]')

        # Ground Truth
        if ground_truth is not None:
            ax.plot3D(ground_truth.pos.north, ground_truth.pos.east, ground_truth.pos.h_asl, linewidth=4, color='r',
                      label='Ground Truth')
            # ax.plot3D(ground_truth.pos.lon, ground_truth.pos.lat, ground_truth.pos.h_asl, linewidth=4, color='r',
            #           label='Ground Truth')

        # Measured
        idx = 10  # every 10th element
        ax.scatter3D(measurements.pos.north[::idx], measurements.pos.east[::idx], measurements.pos.h_asl[::idx],
                     marker='x', color='black', label='Measured')
        # ax.scatter3D(measurements.pos.lon[::idx], measurements.pos.lat[::idx], measurements.pos.h_asl[::idx],
        #              marker='x', color='black', label='Measured')

        # Estimated
        ax.plot3D(estimation_results.traj.pos.north, estimation_results.traj.pos.east, estimation_results.traj.pos.h_asl,
                  linewidth=4, color='b', label='Estimated')
        # ax.plot3D(estimation_results.traj.pos.lon, estimation_results.traj.pos.lat, estimation_results.traj.pos.h_asl,
        #           linewidth=4, color='b', label='Estimated')

        # Legend
        lgd = ax.legend(loc='best')
        lgd.set_title('PATHS')
        lgd.get_frame().set_linewidth(1.0)
        plt.tight_layout()

        # save fig
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        # show fig
        plt.show()
    if args.plots['position errors']:
        title = 'Position Errors'
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle(title, fontsize=24, fontweight='bold')

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
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        # plot show
        plt.show()
    if args.plots['velocity errors']:
        title = 'Velocity Errors'
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(title, fontsize=24, fontweight='bold')
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
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        plt.show()
    if args.plots['altitude errors']:
        title = 'Altitude Errors'
        plt.figure(title, figsize=(10, 12))
        plt.suptitle(title, fontsize=24, fontweight='bold')

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
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        # show plot
        plt.show()
    if args.plots['attitude errors']:
        title = 'Attitude Errors'
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(title, fontsize=24, fontweight='bold')

        # Error in euler psi
        axs[0].plot(args.time_vec, errors.euler.psi, '-r', linewidth=1)
        axs[0].plot(args.time_vec, covars.euler.psi, '--b', linewidth=1)
        axs[0].plot(args.time_vec, -covars.euler.psi, '--b', linewidth=1)
        axs[0].set_title(r'Euler $\psi$ - roll Error [deg]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in euler theta
        axs[1].plot(args.time_vec, errors.euler.theta, '-r', linewidth=1)
        axs[1].plot(args.time_vec, covars.euler.theta, '--b', linewidth=1)
        axs[1].plot(args.time_vec, -covars.euler.theta, '--b', linewidth=1)
        axs[1].set_title(r'Euler $\theta$ - pitch Error [deg]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in euler phi
        axs[2].plot(args.time_vec, errors.euler.phi, '-r', linewidth=1)
        axs[2].plot(args.time_vec, covars.euler.phi, '--b', linewidth=1)
        axs[2].plot(args.time_vec, -covars.euler.phi, '--b', linewidth=1)
        axs[2].set_title(r'Euler $\phi$ - roll Error [m]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].grid(True)
        axs[2].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        plt.tight_layout()

        # save fig
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        # plot show
        plt.show()
    if args.plots['model errors']:
        title = 'Model Errors'
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle(title, fontsize=24, fontweight='bold')

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
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        # show plot
        plt.show()
    if args.plots['kalman gains']:
        title = "Kalman Gains"
        gains = [
            'Position North',
            'Position East',
            'Position Down',
            'Velocity North',
            'Velocity East',
            'Velocity Down'
        ]

        fig, axs = plt.subplots(2, 3, figsize=(10, 12))
        fig.suptitle(title, fontsize=24, fontweight='bold')

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
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

        # plot show
        plt.show()
    if args.plots['map elevation']:
        title = 'Map Elevation'
        fig = plt.figure('Map  - Ground Elevation at PinPoint', figsize=(10, 12))
        fig.suptitle(title, fontsize=24, fontweight='bold')
        plt.plot(args.time_vec, ground_truth.pinpoint.h_map, 'r')
        plt.plot(args.time_vec, estimation_results.traj.pos.h_asl - estimation_results.traj.pos.h_agl, '--b')
        plt.title('Map elevation')
        plt.title('Height [m]')
        plt.xlabel('Time [sec]')
        plt.grid(True)
        plt.legend(['True', 'Estimated'])

        plt.tight_layout()
        # save fig
        plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
        plt.savefig(os.path.join(args.results_folder, f'{title}.svg'))

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
