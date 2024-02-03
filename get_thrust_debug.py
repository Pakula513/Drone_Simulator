import numpy as np
import pandas as pd
import argparse
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description = 'Extract thrust measurement from drone without effects of gravity')
parser.add_argument('--path', dest = 'fpath')
args = parser.parse_args()
filepath = args.fpath

if not filepath:
  filepath = "drone_test_10-2023.csv"




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

timeArray = timeArray / 1000000

sos = signal.butter(7, 10, fs = 500, output = 'sos')
accelArray[:,0] = signal.sosfilt(sos, accelArray[:,0])
accelArray[:,1] = signal.sosfilt(sos, accelArray[:,1])
accelArray[:,2] = signal.sosfilt(sos, accelArray[:,2])
gyroArray[:,0] = signal.sosfilt(sos, gyroArray[:,0])
gyroArray[:,1] = signal.sosfilt(sos, gyroArray[:,1])
gyroArray[:,2] = signal.sosfilt(sos, gyroArray[:,2])



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
    # q_unpacked = q + dt * p
    dot_q = 0.5 * quat_mult(q, p)
    q_unpacked = q + dt * dot_q
    q_unpacked /= np.linalg.norm(q_unpacked)
    print(dt)
    return q_unpacked



t_start = 0.0
q_list = [[t_start, 0.0, 0.0, 0.0, 1.0]] # Unit quaternion
gyro_list = np.hstack((timeArray, gyroArray)) # Load from data, first column time, 2,3,4th w_x, w_y, w_z
accel_i_list = []


err_list = []
counter = 0

for gyro in gyro_list:
    q = q_list[-1]
    dt = gyro[0] - q[0]
    w_applied = gyro[1:]

    
    q_scalar_first = np.array([q[4], *q[1:4]])
    q_next_scalar_first = integrate_quaternion(q_scalar_first, w_applied, dt)  
    # q_next_scalar_first = q_scalar_first + dt * np.array([0, *w_applied])
    q_next = np.array([*q_next_scalar_first[1:4], q_next_scalar_first[0]])
    q_list.append([gyro[0], *q_next])

    # err_list.append(q)
    # R_b = R.from_quat(q_list[-1][1:]).as_matrix()   # Multiply this with measured acceleration to get the original acceleration vector
    # accel_i_list.append(R_b @ accelArray[counter])
    counter += 1

# print(err_list[0])
err_array = np.array(err_list)
# plt.plot(err_array[:,0])
q_arr = np.array(q_list)

plt.plot(q_arr[:,1])
plt.plot(q_arr[:,2])
plt.plot(q_arr[:,3])
plt.plot(q_arr[:,4])
plt.show()
exit()

accel_i_array = np.array(accel_i_list)




# plt.plot(np.arange(accel_i_array[:,0].size), accel_i_array[:,0])
# plt.plot(np.arange(accel_i_array[:,1].size), accel_i_array[:,1])
# plt.plot(np.arange(accel_i_array[:,2].size), accel_i_array[:,2])

l1, l2, l3, l4, l5 = [], [], [], [], []

for item in err_list:
    l1.append(item[0])
    l2.append(item[1])
    l3.append(item[2])
    l4.append(item[3])
    l5.append(item[4])  

a1 = np.array(l1)
a2 = np.array(l2)
a3 = np.array(l3)
a4 = np.array(l4)
a5 = np.array(l5)

b = np.arange(a1.size)
plt.plot(b, a1)
plt.plot(b, a2)
plt.plot(b, a3)
plt.plot(b, a4)
plt.plot(b, a5)


plt.show()

