import matplotlib.pyplot as plt
import numpy as np
import os


# noinspection DuplicatedCode
def plot_results(args, map_data, true_traj, measured, est, errors, plots):
    os.makedirs('Results', exist_ok=True)
    repr(plots)

    if plots['plot map']:
        fig = plt.figure('results_on_map', figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('MAP', fontsize=24, fontweight='bold')
        X, Y = np.meshgrid(map_data.Lon, map_data.Lat)
        ax.plot_surface(X, Y, map_data.map_grid, cmap='bone')
        plt.grid(False)
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_zlabel('Height [m]')

        # Ground Truth
        if true_traj is not None:
            ax.plot3D(true_traj.Lon, true_traj.Lat, true_traj.H_asl, linewidth=4, color='r', label='Ground Truth')

        # Measured
        idx = 10  # every 10th element
        ax.scatter3D(measured.Lon[::idx], measured.Lat[::idx], measured.H_asl[::idx],
                     marker='x', color='black', label='Measured')

        # Estimated
        ax.plot3D(est.est_traj.Lon, est.est_traj.Lat, est.est_traj.H_asl, linewidth=4, color='b', label='Estimated')

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
    if plots['position errors']:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Position Errors', fontsize=24, fontweight='bold')

        # Error in North position
        axs[0].plot(args.time_vec, errors.pos_North, '-r', linewidth=1)
        axs[0].plot(args.time_vec, np.sqrt(errors.cov_pos_north), '--b', linewidth=1)
        axs[0].plot(args.time_vec, -np.sqrt(errors.cov_pos_north), '--b', linewidth=1)
        axs[0].set_title('North Position Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Err', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in East Position
        axs[1].legend(['Err', r'+$\sigma$', r'-$\sigma$'], loc='best')
        # save fig
        plt.savefig(os.path.join(args.results_folder, 'position_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'position_errors.svg'))

        # plot show
        plt.show()
    if plots['velocity errors']:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Velocity Errors', fontsize=24, fontweight='bold')

        # Error in North Velocity
        axs[0].plot(args.time_vec, errors.vel_North, '-r', linewidth=1)
        axs[0].plot(args.time_vec, np.sqrt(errors.cov_vel_north), '--b', linewidth=1)
        axs[0].plot(args.time_vec, -np.sqrt(errors.cov_vel_north), '--b', linewidth=1)
        axs[0].set_title('North Velocity Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in East Velocity
        axs[1].plot(args.time_vec, errors.vel_East, '-r', linewidth=1)
        axs[1].plot(args.time_vec, np.sqrt(errors.cov_vel_east), '--b', linewidth=1)
        axs[1].plot(args.time_vec, -np.sqrt(errors.cov_vel_east), '--b', linewidth=1)
        axs[1].set_title('East Position Error [m]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in Down Velocity
        axs[2].plot(args.time_vec, errors.vel_Down, '-r', linewidth=1)
        axs[2].plot(args.time_vec, np.sqrt(errors.cov_vel_down), '--b', linewidth=1)
        axs[2].plot(args.time_vec, -np.sqrt(errors.cov_vel_down), '--b', linewidth=1)
        axs[2].set_title('East Position Error [m]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].grid(True)
        axs[2].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        plt.tight_layout()

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'velocity_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'velocity_errors.svg'))

        plt.show()
    if plots['altitude errors']:
        plt.figure('Altitude Errors', figsize=(10, 12))
        plt.suptitle('Altitude Errors', fontsize=24, fontweight='bold')

        plt.plot(args.time_vec, errors.pos_alt, 'r', linewidth=1)
        plt.plot(args.time_vec, np.sqrt(errors.cov_pos_down), '--b', linewidth=1)
        plt.plot(args.time_vec, -np.sqrt(errors.cov_pos_down), '--b', linewidth=1)

        mean_err_alt = np.mean(errors.pos_alt)
        plt.plot(args.time_vec, mean_err_alt * np.ones(len(errors.pos_alt)), '*-k')
        plt.plot(args.time_vec, est.params.Z, '-.g')

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
    if plots['attitude errors']:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Attitude Errors', fontsize=24, fontweight='bold')

        # Error in euler psi
        axs[0].plot(args.time_vec, errors.eul_psi, '-r', linewidth=1)
        axs[0].plot(args.time_vec, np.sqrt(errors.cov_eul_psi), '--b', linewidth=1)
        axs[0].plot(args.time_vec, -np.sqrt(errors.cov_eul_psi), '--b', linewidth=1)
        axs[0].set_title('Euler Psi Error [deg]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in euler theta
        axs[1].plot(args.time_vec, errors.eul_Theta, '-r', linewidth=1)
        axs[1].plot(args.time_vec, np.sqrt(errors.cov_eul_theta), '--b', linewidth=1)
        axs[1].plot(args.time_vec, -np.sqrt(errors.cov_eul_theta), '--b', linewidth=1)
        axs[1].set_title('Euler Theta Error [deg]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Error', r'+$\sigma$', r'-$\sigma$'], loc='best')

        # Error in euler phi
        axs[2].plot(args.time_vec, errors.eul_phi, '-r', linewidth=1)
        axs[2].plot(args.time_vec, np.sqrt(errors.eul_phi), '--b', linewidth=1)
        axs[2].plot(args.time_vec, -np.sqrt(errors.eul_phi), '--b', linewidth=1)
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
    if plots['model errors']:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Model Errors', fontsize=24, fontweight='bold')

        axs[0].set_title('Measurement mismatch and Error correction')
        axs[0].plot(args.time_vec, est.params.dX[2, :], '-r')
        axs[0].plot(args.time_vec, est.params.Z, '--b')
        axs[0].set_ylabel('Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[0].grid(True)
        axs[0].legend(['Error', 'Mismatch'], loc='best')

        axs[1].set_title('Process Noise, R and Rc')
        axs[1].plot(args.time_vec, est.params.Rc, '-r')
        axs[1].plot(args.time_vec, est.params.Rfit, '--b')
        # axs[1].scatter(args.time_vec[::10], est.params.R[::10], '*g')
        axs[0].set_ylabel('Error [m]')
        axs[0].set_xlabel('Time [sec]')
        axs[1].grid(True)
        axs[1].legend(['Rc - penalty on height', 'Rfit', 'R'], loc='best')

        # save fig
        plt.savefig(os.path.join(args.results_folder, 'model_errors.png'))
        plt.savefig(os.path.join(args.results_folder, 'model_errors.svg'))

        # show plot
        plt.show()
    if plots['kalman gains']:
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

            axs[row, col].plot(args.time_vec, est.params.K[i, :], linewidth=1)
            axs[row, col].set_title(title)
            axs[row, col].set_xlabel('Time [sec]')
            axs[row, col].grid(True)

        plt.tight_layout()
        # save fig
        plt.savefig(os.path.join(args.results_folder, 'kalman_gains.png'))
        plt.savefig(os.path.join(args.results_folder, 'kalman_gains.svg'))

        # plot show
        plt.show()
    if plots['map elevation']:

        fig = plt.figure('Map  - Ground Elevation at PinPoint', figsize=(10, 12))
        fig.suptitle('Ground Elevation at PinPoint', fontsize=24, fontweight='bold')
        plt.plot(args.time_vec, true_traj.pinpoint.H_map, 'r')
        plt.plot(args.time_vec, est.est_traj.H_asl - est.est_traj.H_agl, '--b')
        plt.title('Map Altitude [m]')
        plt.xlabel('Time [sec]')
        plt.grid(True)
        plt.legend(['True', 'Estimated'])

        plt.tight_layout()
        # save fig
        plt.savefig(os.path.join(args.results_folder, 'map_elevation.png'))
        plt.savefig(os.path.join(args.results_folder, 'map_elevation.svg'))

        # show plot
        plt.show()
