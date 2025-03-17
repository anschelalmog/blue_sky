import os
import argparse
import numpy as np

def set_settings():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--traj_from_file', type=bool, default=False,
                        help='load the trajectory from file or to generate one')
    parser.add_argument('--errors_from_table', type=bool, default=True,
                        help='load the errors from file or to generate one')
    parser.add_argument('--traj_path', type=str, default='', help='the path of the trajectory file')
    parser.add_argument('--plot_results', type=bool, default=True,
                        help='plot and save plots results at the end of the run')
    parser.add_argument('--noise_type', type=str, default='normal',
                        help='measurements noise type: none, normal or uniform')

    # Map Settings
    parser.add_argument('--maps_dir', type=str, default='Map',
                        help='Path to maps, format: "Map/LevelX/DTED/E0XX/mXX.mat".')
    parser.add_argument('--map_level', type=int, default=1, help='map level')

    # Time Settings
    parser.add_argument('--time_init', type=int, default=0, help='times starts counting, in [sec]')
    parser.add_argument('--time_end', type=int, default=100, help='times ends counting, in [sec]')
    # parser.add_argument('--time_rate', type=float, default=1, help='time rate, in [sec]')
    parser.add_argument('--time_res', type=float, default=0.1, help='flights resolution speed, in [sec]')

    # Kalman Filter Settings
    parser.add_argument('--kf_type', type=str, default='IEKF', help='kalman filter type, format: IEKF or UKF')
    # dX = [ΔPos_North, ΔPos_East, ΔH_asl, ΔVel_North, ΔVel_East, ΔVel_Down, ΔAcc_North, ΔAcc_East, ΔAcc_Down,
    #                                                                           Δψ, Δθ, Δφ] space state vector
    parser.add_argument('--kf_state_size', type=int, default=12, help='number of state estimation')

    args = parser.parse_args()

    if args.errors_from_table:
        parser.add_argument('--imu_errors', type=dict, default=None, help='IMU error parameters')
    else:
        # Errors Flags
        parser.add_argument('--flg_err_pos', type=bool, default=False, help='flag error for position')
        parser.add_argument('--flg_err_vel', type=bool, default=False, help='flag for error for velocity')
        parser.add_argument('--flg_err_alt', type=bool, default=False, help='flag error for altimeter')
        parser.add_argument('--flg_err_acc', type=bool, default=False, help='flag error for accelerometer')
        parser.add_argument('--flg_err_eul', type=bool, default=False, help='flag error for euler angels')
        parser.add_argument('--flg_err_baro_noise', type=bool, default=False, help='flag error for barometer noise')
        parser.add_argument('--flg_err_baro_bias', type=bool, default=False, help='flag error for barometer bias')

        # Errors Values
        parser.add_argument('--val_err_pos', type=float, default=200, help='error for position, in [m]')
        parser.add_argument('--val_err_vel', type=float, default=2, help='error for velocity, in [m/s]')
        parser.add_argument('--val_err_alt', type=float, default=5, help='error for altimeter, in [m]')
        parser.add_argument('--val_err_acc', type=float, default=5, help='error for  accelerometer, in [m/s^2]')
        parser.add_argument('--val_err_eul', type=float, default=0.05, help='error for euler angels, in [deg]')
        parser.add_argument('--val_err_baro_noise', type=float, default=5, help='error for barometer, in [m]')
        parser.add_argument('--val_err_baro_bias', type=float, default=5, help='error for barometer, in [m]')
        parser.imu_errors = {
            'velocity': args.flg_err_vel * args.val_err_vel,
            'initial_position': args.flg_err_pos * args.val_err_pos,
            'euler_angles': args.flg_err_eul * args.val_err_eul,
            'barometer_bias': args.flg_err_baro_bias * args.val_err_baro_bias,
            'barometer_noise': args.flg_err_baro_noise * args.val_err_baro_noise,
            'altimeter_noise': args.flg_err_alt * args.val_err_alt,
            'accelerometer': args.flg_err_acc * args.val_err_acc
        }

    # Flight Settings
    if not args.traj_from_file:
        parser.add_argument('--init_lat', type=float, default=37.5, help='initial Latitude, in [deg]')
        parser.add_argument('--init_lon', type=float, default=21.5, help='initial Longitude, in [deg]')
        parser.add_argument('--init_height', type=float, default=5000, help='flight height at start, in [m]')
        #
        parser.add_argument('--avg_spd', type=float, default=250, help='flight average speed, [in m/sec]')
        parser.add_argument('--psi', type=float, default=45, help='Yaw at start, in [deg]')
        parser.add_argument('--theta', type=float, default=0, help='Pitch at start, in [deg]')
        parser.add_argument('--phi', type=float, default=0, help='Roll at start, in [deg]')
        #
        parser.add_argument('--acc_north', type=float, default=0, help='acceleration in the north - south at start, '
                                                                       'in [m/s^2]')
        parser.add_argument('--acc_east', type=float, default=0, help='acceleration in the east - west axis at start, '
                                                                      'in [m/s^2]')
        parser.add_argument('--acc_down', type=float, default=0, help='acceleration in vertical axis at start, '
                                                                      'in [m/s^2]')
        #
        parser.add_argument('--psi_dot', type=float, default=0, help='change in psi during flight, '
                                                                     'in [deg/s]')
        parser.add_argument('--theta_dot', type=float, default=0, help='change in theta during flight, '
                                                                       'in [deg/s]')
        parser.add_argument('--phi_dot', type=float, default=0, help='change in phi during flight, '
                                                                     'in [deg/s]')
    else:  # already read from file
        pass

    # Other Defaults
    args = parser.parse_args()
    args.run_points = int(args.time_end / args.time_res)
    args.time_vec = np.arange(args.time_init, args.time_end, args.time_res)
    args.map_res = 3 if args.map_level == 1 else 1
    args.results_folder = os.path.join(os.getcwd(), 'out')

    return args
