import numpy as np
import pandas as pd
import argparse
from scipy import signal
from scipy.spatial.transform import Rotation as R



parser = argparse.ArgumentParser(description = 'Extract thrust measurement from drone without effects of gravity')
parser.add_argument('--path', dest = 'fpath')
args = parser.parse_args()
filepath = args.fpath

if not filepath:
  filepath = "drone_test_1_BB.csv"




def load_filter_bf_datafile(signal_order, signal_crit_freq, signal_sampling_freq, signal_output):

    # rcCommand 0 is roll, 1 is pitch, 2 is yaw
    blackbox_df = pd.read_csv(filepath, skiprows = 141, 
                            usecols = ['time', 'rcCommand[0]', 'rcCommand[1]', 'rcCommand[2]', 
                                       'gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]', 
                                       'accSmooth[0]', 'accSmooth[1]', 'accSmooth[2]', 
                                       'motor[0]', 'motor[1]', 'motor[2]', 'motor[3]'])
    

    pitchArray = np.array(blackbox_df.loc[:, 'rcCommand[1]'].to_numpy(), ndmin = 2).T
    rollArray = np.array(blackbox_df.loc[:, 'rcCommand[0]'].to_numpy(), ndmin = 2).T
    yawArray = np.array(blackbox_df.loc[:, 'rcCommand[2]'].to_numpy(), ndmin = 2).T
    stickArray = np.hstack((pitchArray, rollArray, yawArray))

    timeArray = np.array(blackbox_df.loc[:, 'time'].to_numpy(), ndmin = 2).T
    timeArray -= 20270187



    gyro_x = np.array(blackbox_df.loc[:, 'gyroADC[0]'].to_numpy(), ndmin = 2).T
    gyro_y = np.array(blackbox_df.loc[:, 'gyroADC[1]'].to_numpy(), ndmin = 2).T
    gyro_z = np.array(blackbox_df.loc[:, 'gyroADC[2]'].to_numpy(), ndmin = 2).T


    sos = signal.butter(signal_order, signal_crit_freq, fs = signal_sampling_freq, output = signal_output)
    filtered = signal.sosfilt(sos, gyro_x)
    filtered2 = signal.sosfilt(sos, gyro_y)
    filtered3 = signal.sosfilt(sos, gyro_z)


    filtered_gyroArray = np.hstack((filtered, filtered2, filtered3))

    accel_0 = np.array(blackbox_df.loc[:, 'accSmooth[0]'].to_numpy(), ndmin=2).T / 2048
    accel_1 = np.array(blackbox_df.loc[:, 'accSmooth[1]'].to_numpy(), ndmin=2).T / 2048
    accel_2 = np.array(blackbox_df.loc[:, 'accSmooth[2]'].to_numpy(), ndmin=2).T / 2048

    accelArray = np.hstack((accel_0, accel_1, accel_2))


    motor_0 = np.array(blackbox_df.loc[:, 'motor[0]'].to_numpy(), ndmin=2).T
    motor_1 = np.array(blackbox_df.loc[:, 'motor[1]'].to_numpy(), ndmin=2).T
    motor_2 = np.array(blackbox_df.loc[:, 'motor[2]'].to_numpy(), ndmin=2).T
    motor_3 = np.array(blackbox_df.loc[:, 'motor[3]'].to_numpy(), ndmin=2).T

    motorArray = np.hstack((motor_0, motor_1, motor_2, motor_3))



    return stickArray, filtered_gyroArray, accelArray, motorArray, timeArray


stickArray, gyroArray, accelArray, motorArray, timeArray = load_filter_bf_datafile(7, 100, 500, 'sos')









# Scalar first
def quat_mult(q, p):
    r = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r[0] = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
    r[1] = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
    r[2] = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1]
    r[3] = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
    return r

# Scalar first
# Forward eular, could do trapezoidal
def integrate_quaternion(q, gyr, dt):
    p = np.array([0.0, gyr[0], gyr[1], gyr[2]], dtype=np.float32)
    dot_q = 0.5 * quat_mult(q, p)
    q_unpacked = q + dt * dot_q
    q_unpacked /= np.linalg.norm(q_unpacked)
    return q_unpacked

# Scalar last

t_start = 0.0
q_list = [t_start, 0.0, 0.0, 0.0, 1.0] # Unit quaternion
gyro_list = [] # TODO Load from data, first column time, 2,3,4th w_x, w_y, w_z   (gyrox, gyroy, gyroz)

for gyro in gyro_list:
    q = q_list[-1]
    dt = gyro[0] - q[0]
    w_applied = gyro[1:]

    q_scalar_first = np.array([q[4], *q[1:4]])
    q_next_scalar_first = integrate_quaternion(q_scalar_first, w_applied, dt)   
    q_next = np.array([*q_next_scalar_first[1:4], q_next_scalar_first[0]])
    q_list.append([gyro[0], *q_next])

    R_ib = R.from_quat(q_list[-1][1:]).as_matrix()   # Multiply this with measured acceleration to get the original acceleration vector