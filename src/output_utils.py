import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from src.base_traj import BaseTraj
from src.noise_traj import NoiseTraj
import numpy.random as rnd


# noinspection DuplicatedCode


class RunErrors(BaseTraj):
    def __init__(self, used_traj, est_traj, covars):
        super().__init__(used_traj.run_points)
        self.pos.north = used_traj.pos.north - est_traj.pos.lat * used_traj.mpd_north
        self.pos.east = (used_traj.pos.lon - est_traj.pos.lon)  # used_traj.mpd_east
        self.pos.h_asl = used_traj.pos.h_asl - est_traj.pos.h_asl
        #
        self.vel.north = used_traj.vel.north - est_traj.vel.north
        self.vel.east = used_traj.vel.east - est_traj.vel.east
        self.vel.down = used_traj.vel.down - est_traj.vel.down
        #
        self.euler.psi = used_traj.euler.psi - est_traj.euler.psi
        self.euler.theta = used_traj.euler.theta - est_traj.euler.theta
        self.euler.phi = used_traj.euler.phi - est_traj.euler.phi
        #

        self.metrics = {
            'pos': {
                'north': {
                    'rmse': np.sqrt(np.mean(self.pos.north ** 2)),
                    'max_abs_error': np.max(np.abs(self.pos.north)),
                    'error_bound_percentage': np.mean(np.abs(self.pos.north) <= np.abs(covars.pos.north)) * 100
                },
                'east': {
                    'rmse': np.sqrt(np.mean(self.pos.east ** 2)),
                    'max_abs_error': np.max(np.abs(self.pos.east)),
                    'error_bound_percentage': np.mean(np.abs(self.pos.east) <= np.abs(covars.pos.east)) * 100
                },
                'h_asl': {
                    'rmse': np.sqrt(np.mean(self.pos.h_asl ** 2)),
                    'max_abs_error': np.max(np.abs(self.pos.h_asl)),
                    'error_bound_percentage': np.mean(np.abs(self.pos.h_asl) <= np.abs(covars.pos.h_asl)) * 100
                }
            },
            'vel': {
                'north': {
                    'rmse': np.sqrt(np.mean(self.vel.north ** 2)),
                    'max_abs_error': np.max(np.abs(self.vel.north)),
                    'error_bound_percentage': np.mean(np.abs(self.vel.north) <= np.abs(covars.vel.north)) * 100
                },
                'east': {
                    'rmse': np.sqrt(np.mean(self.vel.east ** 2)),
                    'max_abs_error': np.max(np.abs(self.vel.east)),
                    'error_bound_percentage': np.mean(np.abs(self.vel.east) <= np.abs(covars.vel.east)) * 100
                },
                'down': {
                    'rmse': np.sqrt(np.mean(self.vel.down ** 2)),
                    'max_abs_error': np.max(np.abs(self.vel.down)),
                    'error_bound_percentage': np.mean(np.abs(self.vel.down) <= np.abs(covars.vel.down)) * 100
                }
            },
            'euler': {
                'psi': {
                    'rmse': np.sqrt(np.mean(self.euler.psi ** 2)),
                    'max_abs_error': np.max(np.abs(self.euler.psi)),
                    'error_bound_percentage': np.mean(np.abs(self.euler.psi) <= np.abs(covars.euler.psi)) * 100
                },
                'theta': {
                    'rmse': np.sqrt(np.mean(self.euler.theta ** 2)),
                    'max_abs_error': np.max(np.abs(self.euler.theta)),
                    'error_bound_percentage': np.mean(np.abs(self.euler.theta) <= np.abs(covars.euler.theta)) * 100
                },
                'phi': {
                    'rmse': np.sqrt(np.mean(self.euler.phi ** 2)),
                    'max_abs_error': np.max(np.abs(self.euler.phi)),
                    'error_bound_percentage': np.mean(np.abs(self.euler.phi) <= np.abs(covars.euler.phi)) * 100
                }
            }
        }


class Covariances(BaseTraj):
    def __init__(self, covariances, traj):
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
        self.acc.north = np.sqrt(covariances[6, 6, :])
        self.acc.east = np.sqrt(covariances[7, 7, :])
        self.acc.down = np.sqrt(covariances[8, 8, :])
        #
        self.euler.psi = np.sqrt(covariances[9, 9, :])
        self.euler.theta = np.sqrt(covariances[10, 10, :])
        self.euler.phi = np.sqrt(covariances[11, 11, :])

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
    x = 1000
    if args.plots['plot map']:
        title = 'Results on Map'
        fig = plt.figure(title, figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('MAP', fontsize=24, fontweight='bold')
        X, Y = np.meshgrid(map_data.axis['north'], map_data.axis['east'])
        ax.plot_surface(X, Y, map_data.grid, cmap='bone')
        plt.grid(False)
        ax.set_xlabel('North [m]')
        ax.set_ylabel('East [m]')
        ax.set_zlabel('Height [m]')

        # # Ground Truth
        # if ground_truth is not None:
        #     ax.plot3D(ground_truth.pos.north, ground_truth.pos.east, ground_truth.pos.h_asl, linewidth=4, color='r',
        #               label='Ground Truth')
        #     # ax.plot3D(ground_truth.pos.lon, ground_truth.pos.lat, ground_truth.pos.h_asl, linewidth=4, color='r',
        #     #           label='Ground Truth')

        # Measured
        idx = 10  # every 10th element
        ax.scatter3D(measurements.pos.north[::idx], measurements.pos.east[::idx], measurements.pos.h_asl[::idx],
                     marker='x', color='black', label='Measured')
        # ax.scatter3D(measurements.pos.lon[::idx], measurements.pos.lat[::idx], measurements.pos.h_asl[::idx],
        #              marker='x', color='black', label='Measured')

        # # Estimated
        # ax.plot3D(estimation_results.traj.pos.north, estimation_results.traj.pos.east,
        #           estimation_results.traj.pos.h_asl,
        #           linewidth=4, color='b', label='Estimated')
        # # ax.plot3D(estimation_results.traj.pos.lon, estimation_results.traj.pos.lat, estimation_results.traj.pos.h_asl,
        # #           linewidth=4, color='b', label='Estimated')

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
        axs[0].plot(args.time_vec[:x], errors.pos.north[:x], '-r', linewidth=1)
        axs[0].plot(args.time_vec[:x], covars.pos.north[:x], '--b', linewidth=1)
        axs[0].plot(args.time_vec[:x], -covars.pos.north[:x], '--b', linewidth=1)
        axs[0].set_title('North Position Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'$\pm\sigma$'], loc='upper right')

        # Add error metrics as text on the North Position subplot
        rmse_north = errors.metrics['pos']['north']['rmse']
        max_abs_error_north = errors.metrics['pos']['north']['max_abs_error']
        error_bound_percentage_north = errors.metrics['pos']['north']['error_bound_percentage']
        axs[0].text(0.05, 0.95,
                    f"RMSE: {rmse_north:.4f}\nMax Abs Error: {max_abs_error_north:.4f}\nError Bound Percentage: {error_bound_percentage_north:.2f}%",
                    transform=axs[0].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Error in East Position
        axs[1].plot(args.time_vec[:x], errors.pos.east[:x], '-r', linewidth=1)
        axs[1].plot(args.time_vec[:x], covars.pos.east[:x], '--b', linewidth=1)
        axs[1].plot(args.time_vec[:x], -covars.pos.east[:x], '--b', linewidth=1)
        axs[1].set_title('East Position Error [m]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'+$\pm\sigma$'], loc='upper right')

        # Add error metrics as text on the East Position subplot
        rmse_east = errors.metrics['pos']['east']['rmse']
        max_abs_error_east = errors.metrics['pos']['east']['max_abs_error']
        error_bound_percentage_east = errors.metrics['pos']['east']['error_bound_percentage']
        axs[1].text(0.05, 0.95,
                    f"RMSE: {rmse_east:.4f}\nMax Abs Error: {max_abs_error_east:.4f}\nError Bound Percentage: {error_bound_percentage_east:.2f}%",
                    transform=axs[1].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

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
        axs[0].set_title('North Velocity Error [m/s]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'$\pm\sigma$'], loc='upper right')

        # Error in East Velocity
        axs[1].plot(args.time_vec, errors.vel.east, '-r', linewidth=1)
        axs[1].plot(args.time_vec, covars.vel.east, '--b', linewidth=1)
        axs[1].plot(args.time_vec, -covars.vel.east, '--b', linewidth=1)
        axs[1].set_title('East Velocity Error [m/s]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'$\pm\sigma$'], loc='upper right')

        # Error in Down Velocity
        axs[2].plot(args.time_vec, errors.vel.down, '-r', linewidth=1)
        axs[2].plot(args.time_vec, covars.vel.down, '--b', linewidth=1)
        axs[2].plot(args.time_vec, -covars.vel.down, '--b', linewidth=1)
        axs[2].set_title('Down Velocity Error [m/s]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].grid(True)
        axs[2].legend(['Error', r'$\pm\sigma$'], loc='upper right')

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

        # Add error metrics as text on the Altitude Errors plot
        rmse_alt = errors.metrics['pos']['h_asl']['rmse']
        max_abs_error_alt = errors.metrics['pos']['h_asl']['max_abs_error']
        error_bound_percentage_alt = errors.metrics['pos']['h_asl']['error_bound_percentage']
        plt.text(0.05, 0.95,
                 f"RMSE: {rmse_alt:.4f}\nMax Abs Error: {max_abs_error_alt:.4f}\nError Bound Percentage: {error_bound_percentage_alt:.2f}%",
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

        # legend
        plt.title('Altitude Err [m]')
        plt.xlabel('Time [sec]')
        plt.grid(True)
        plt.legend(['Error', r'$\pm\sigma$', 'mean', 'Z'], loc='upper right')

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
        axs[0].set_title(r'Euler $\psi$ - yaw Error [deg]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'+$\pm\sigma$'], loc='upper right')

        # Add error metrics as text on the Euler Psi subplot
        rmse_psi = errors.metrics['euler']['psi']['rmse']
        max_abs_error_psi = errors.metrics['euler']['psi']['max_abs_error']
        error_bound_percentage_psi = errors.metrics['euler']['psi']['error_bound_percentage']
        axs[0].text(0.05, 0.95,
                    f"RMSE: {rmse_psi:.4f}\nMax Abs Error: {max_abs_error_psi:.4f}\nError Bound Percentage: {error_bound_percentage_psi:.2f}%",
                    transform=axs[0].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Error in euler theta
        axs[1].plot(args.time_vec, errors.euler.theta, '-r', linewidth=1)
        axs[1].plot(args.time_vec, covars.euler.theta, '--b', linewidth=1)
        axs[1].plot(args.time_vec, -covars.euler.theta, '--b', linewidth=1)
        axs[1].set_title(r'Euler $\theta$ - pitch Error [deg]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'$\pm\sigma$'], loc='upper right')

        # Add error metrics as text on the Euler Theta subplot
        rmse_theta = errors.metrics['euler']['theta']['rmse']
        max_abs_error_theta = errors.metrics['euler']['theta']['max_abs_error']
        error_bound_percentage_theta = errors.metrics['euler']['theta']['error_bound_percentage']
        axs[1].text(0.05, 0.95,
                    f"RMSE: {rmse_theta:.4f}\nMax Abs Error: {max_abs_error_theta:.4f}\nError Bound Percentage: {error_bound_percentage_theta:.2f}%",
                    transform=axs[1].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Error in euler phi
        axs[2].plot(args.time_vec, errors.euler.phi, '-r', linewidth=1)
        axs[2].plot(args.time_vec, covars.euler.phi, '--b', linewidth=1)
        axs[2].plot(args.time_vec, -covars.euler.phi, '--b', linewidth=1)
        axs[2].set_title(r'Euler $\phi$ - roll Error [deg]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].grid(True)
        axs[2].legend(['Error', r'$\pm\sigma$'], loc='upper right')

        # Add error metrics as text on the Euler Phi subplot
        rmse_phi = errors.metrics['euler']['phi']['rmse']
        max_abs_error_phi = errors.metrics['euler']['phi']['max_abs_error']
        error_bound_percentage_phi = errors.metrics['euler']['phi']['error_bound_percentage']
        axs[2].text(0.05, 0.95,
                    f"RMSE: {rmse_phi:.4f}\nMax Abs Error: {max_abs_error_phi:.4f}\nError Bound Percentage: {error_bound_percentage_phi:.2f}%",
                    transform=axs[2].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

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
        #axs[0].set_ylim(-1, 1.1)
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

        log_file.write("\nMetrics:\n")
        print("\nMetrics:")
        for attr_group, attr_dict in covs.metrics.items():
            log_file.write(f"\n{attr_group.upper()}:\n")
            print(f"\n{attr_group.upper()}:")
            for attr_name, metrics in attr_dict.items():
                log_file.write(f"{attr_name}:\n")
                print(f"{attr_name}:")
                for metric_name, value in metrics.items():
                    log_file.write(f"  {metric_name}: {value:.4f}\n")
                    print(f"  {metric_name}: {value:.4f}")


def compare_trajectories(true_traj, meas_traj):
    print("Comparing Trajectory Components:")
    print("=" * 40)

    for attr in ['pos', 'vel', 'acc', 'euler']:
        true_obj = getattr(true_traj, attr)
        meas_obj = getattr(meas_traj, attr)

        print(f"Comparing {attr}:")
        for sub_attr in vars(true_obj):
            true_val = getattr(true_obj, sub_attr)
            meas_val = getattr(meas_obj, sub_attr)

            if not np.allclose(true_val, meas_val):
                print(f"  - {sub_attr} is not equal")
            else:
                print(f"  - {sub_attr} is equal")

        print()

    # Compare mpd_north and mpd_east
    print("Comparing mpd_north and mpd_east:")
    if not np.allclose(true_traj.mpd_north, meas_traj.mpd_north):
        print("  - mpd_north is not equal")
    else:
        print("  - mpd_north is equal")

    if not np.allclose(true_traj.mpd_east, meas_traj.mpd_east):
        print("  - mpd_east is not equal")
    else:
        print("  - mpd_east is equal")

    print("=" * 40)


def plot_height_profiles(true_traj, noise_traj):
    plt.figure(figsize=(12, 10))
    plt.title('Map Height Comparison')
    # plt.plot(true_traj.time_vec, true_traj.pos.h_map, 'b-', label='Noisy Heights')
    plt.scatter(true_traj.time_vec[::5], true_traj.pos.h_map[::5], c='r', s=30, alpha=0.8, label='Noisy Heights')
    plt.plot(noise_traj.time_vec, noise_traj.pos.h_map, 'k-', label='True Heights')
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('map_height_noise_comparison.jpg')
    plt.show()


def plot_trajectory_comparison(true_traj, map_data, amp, drift, bias):
    fig, ax = plt.subplots(figsize=(12, 10))

    vec_shape = true_traj.pos.north.shape
    amp = 100
    drift = 0.5
    bias = 250

    # Generate noise for top-down approach (only amplitude noise)
    top_down_north = true_traj.pos.north + amp * rnd.randn(*vec_shape)
    top_down_east = true_traj.pos.east + amp * rnd.randn(*vec_shape)

    # Generate noise for bottom-up approach (bias, drift, and amplitude noise)
    bottom_up_north = true_traj.pos.north + bias * np.ones(vec_shape)
    bottom_up_east = true_traj.pos.east + bias * np.zeros(vec_shape)

    bottom_up_north += np.cumsum(drift * np.ones(vec_shape), axis=0)
    bottom_up_east += np.cumsum(drift * rnd.randn(*vec_shape), axis=0)

    bottom_up_north += amp * rnd.randn(*vec_shape)
    bottom_up_east += amp * rnd.randn(*vec_shape)

    # Plot setup
    lat_grid, lon_grid = np.meshgrid(map_data.axis['lat'], map_data.axis['lon'], indexing='ij')
    north_grid = lat_grid * map_data.mpd['north']
    east_grid = lon_grid * map_data.mpd['east']

    # Plot setup
    ax.set_title('Trajectory Comparison: Top-Down vs Bottom-Up')
    ax.contourf(east_grid, north_grid, map_data.grid, cmap='bone', alpha=0.6)
    ax.plot(top_down_east, top_down_north, 'b', linestyle=':', linewidth=2, label='Top-Down Trajectory')
    ax.plot(bottom_up_east, bottom_up_north, 'r', linestyle='-.', linewidth=2, label='Bottom-Up Trajectory')
    ax.plot(true_traj.pos.east, true_traj.pos.north, 'w', linestyle='-', linewidth=2, label='True Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()

    # Calculate limits with margin
    min_east = np.min(np.vstack((true_traj.pos.east, top_down_east, bottom_up_east)))
    max_east = np.max(np.vstack((true_traj.pos.east, top_down_east, bottom_up_east)))
    min_north = np.min(np.vstack((true_traj.pos.north, top_down_north, bottom_up_north)))
    max_north = np.max(np.vstack((true_traj.pos.north, top_down_north, bottom_up_north)))

    east_margin = 0.05 * (max_east - min_east)
    north_margin = 0.05 * (max_north - min_north)

    ax.set_xlim([min_east - east_margin, max_east + east_margin])
    ax.set_ylim([min_north - north_margin, max_north + north_margin])

    plt.tight_layout()
    plt.savefig('Trajectory Comparison.jpg')
    plt.show()


def calc_errors_covariances(meas_traj, estimation_results):
    covariances = Covariances(estimation_results.params.P_est, estimation_results.traj)
    errors = RunErrors(meas_traj, estimation_results.traj, covariances)
    return errors, covariances


def plot_pinpoint_trajectories_2d(true_traj, map_data):
    """
    Plots the true trajectory, measured trajectory, and pinpoint results in 2D.

    :param true_traj: True trajectory data containing the real positions of the vehicle.
    :param map_data: The map data containing the grid and axis information.
    """
    idx = 50
    fig2d = plt.figure(figsize=(8, 8))
    ax1 = fig2d.add_subplot(111)
    X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
    ax1.contourf(X, Y, map_data.grid, cmap='bone', alpha=0.5)

    # Plot true trajectory
    ax1.plot(true_traj.pos.lon, true_traj.pos.lat, 'k--', label='True Trajectory')

    # Plot measured trajectory
    measured_lon = true_traj.pos.lon[::idx]
    measured_lat = true_traj.pos.lat[::idx]
    ax1.scatter(measured_lon, measured_lat, color='blue', marker='o', label='Pinpoint Results')

    # Plot pinpoint results
    pinpoint_lon = true_traj.pinpoint.lon[::idx]
    pinpoint_lat = true_traj.pinpoint.lat[::idx]
    ax1.scatter(pinpoint_lon, pinpoint_lat, color='red', marker='x', label='Measured Trajectory')

    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')
    ax1.set_title('PinPoint results in 2D')
    ax1.legend()

    ax1.set_xlim([min(pinpoint_lon) - 0.05, max(pinpoint_lon) + 0.05])
    ax1.set_ylim([min(pinpoint_lat) - 0.05, max(pinpoint_lat) + 0.05])

    plt.tight_layout()
    plt.savefig('pinpoint_2d.jpg')
    plt.show()


def plot_pinpoint_trajectories_3d(true_traj, map_data, pov=(50, 200), zoom=2.3):
    """
    Plots the true trajectory, measured trajectory, and pinpoint results in 3D.

    :param true_traj: True trajectory data containing the real positions of the vehicle.
    :param map_data: The map data containing the grid and axis information.
    :param pov: Point of view for the 3D plot. Should be a tuple (elev, azim).
    :param zoom: Zoom level for the 3D plot. Should be a float greater than 0 (e.g., 1.5 for 150% zoom).
    """
    idx = 50
    fig3d = plt.figure(figsize=(8, 8))
    ax2 = fig3d.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])

    # Plot true trajectory
    ax2.plot(true_traj.pos.lon, true_traj.pos.lat, true_traj.pos.h_asl, 'k--', label='True Trajectory')

    # Plot measured trajectory
    measured_lon = true_traj.pos.lon[::idx]
    measured_lat = true_traj.pos.lat[::idx]
    ground_elevations_measured = [
        map_data.grid[np.abs(map_data.axis['lat'] - lat).argmin(), np.abs(map_data.axis['lon'] - lon).argmin()]
        for lat, lon in zip(measured_lat, measured_lon)
    ]
    ax2.scatter(measured_lon, measured_lat, ground_elevations_measured, color='blue', marker='o',
                label='Pinpoint Results')

    # Plot pinpoint results
    pinpoint_lon = true_traj.pinpoint.lon[::idx]
    pinpoint_lat = true_traj.pinpoint.lat[::idx]
    ground_elevations_pinpoint = [
        map_data.grid[np.abs(map_data.axis['lat'] - lat).argmin(), np.abs(map_data.axis['lon'] - lon).argmin()]
        for lat, lon in zip(pinpoint_lat, pinpoint_lon)
    ]
    ax2.scatter(pinpoint_lon, pinpoint_lat, ground_elevations_pinpoint, color='red', marker='x',
                label='Measured Trajectory')

    # ax2.plot_surface(X, Y, map_data.grid, cmap='bone', alpha=0.3)

    ax2.set_xlabel('Longitude [deg]')
    ax2.set_ylabel('Latitude [deg]')
    ax2.set_zlabel('Altitude [m]')
    ax2.set_title('PinPoint results in 3D')
    ax2.legend()

    # Set the POV if specified
    ax2.view_init(elev=pov[0], azim=pov[1])

    # Set the zoom if specified
    x_mid = (ax2.get_xlim()[0] + ax2.get_xlim()[1]) / 2
    y_mid = (ax2.get_ylim()[0] + ax2.get_ylim()[1]) / 2
    z_mid = (ax2.get_zlim()[0] + ax2.get_zlim()[1]) / 2

    x_range = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / zoom
    y_range = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) / zoom
    z_range = (ax2.get_zlim()[1] - ax2.get_zlim()[0]) / zoom

    ax2.set_xlim([x_mid - x_range / 2, x_mid + x_range / 2])
    ax2.set_ylim([y_mid - y_range / 2, y_mid + y_range / 2])
    ax2.set_zlim([z_mid - z_range / 2, z_mid + z_range / 2])

    plt.tight_layout()
    plt.savefig('pinpoint_3d.jpg')
    plt.show()
