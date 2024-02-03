import numpy as np
import pandas as pd
import argparse
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import mujoco
import traceback
import time



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
    dot_q = 0.5 * quat_mult(q, p)
    q_unpacked = q + dt * dot_q
    q_unpacked /= np.linalg.norm(q_unpacked)
    return q_unpacked

# Scalar last
err_list = []







def create_accelArray_rotated(gyro_list, accel_Array):
  t_start = 0.0
  q_list = [[t_start, 0.0, 0.0, 0.0, 1.0]] # Unit quaternion
  accel_i_list = []
  counter = 0

  for gyro in gyro_list:
      q = q_list[-1]
      dt = gyro[0] - q[0]
      w_applied = gyro[1:]
      
      q_scalar_first = np.array([q[4], *q[1:4]])
      q_next_scalar_first = integrate_quaternion(q_scalar_first, w_applied, dt)  
      q_next = np.array([*q_next_scalar_first[1:4], q_next_scalar_first[0]])
      q_list.append([gyro[0], *q_next])

      R_b = R.from_quat(q_list[-1][1:]).as_matrix()   # Multiply this with measured acceleration to get the original acceleration vector
      accel_i_list.append(R_b @ accel_Array[counter])
      counter += 1

  return np.array(accel_i_list)


gyro_list_1 = np.hstack((timeArray, gyroArray)).tolist() # Load from data, first column time, 2,3,4th w_x, w_y, w_z
accel_i_array = create_accelArray_rotated(gyro_list_1, accelArray)



# plt.plot(np.arange(accel_i_array[:,0].size), accel_i_array[:,0])
# plt.plot(np.arange(accel_i_array[:,1].size), accel_i_array[:,1])
# plt.plot(np.arange(accel_i_array[:,2].size), accel_i_array[:,2])
# plt.show()

# exit()










accel_2_list = []
gyro_2_list = []
force_2_list = []
actuator_arr = np.array([[ 1,-1, 1, 1],
                         [ 1, 1,-1, 1],
                         [-1, 1, 1, 1],
                         [-1,-1,-1, 1]])

m = mujoco.MjModel.from_xml_path('hoopflyt_noground.xml')
d = mujoco.MjData(m)
prev_error = np.array((-1000,-1000,-1000))
gyro_integ = np.zeros((3,))
optimize = [1.32888640e-01, 9.20171364e-02, 7.46626638e-02, 2.67066373e-01,
            4.98975763e-01, 3.98492489e-01, 5.76530949e-03, 2.79339024e-04,
            2.61102594e-02, 4.01023842e-01, 3.43738627e-01, 7.83487960e-01]


def drone_control(optimize, m, d, stickArray, previous_error, gyro_integ):

  #optimize the following variables in optimize: [Kp, Kp, Kp, Kd, Kd, Kd, Ki, Ki, Ki, input gain, input gain, input gain]  
  try:
    

    Kp = np.array(optimize[0:3])
    Kd = np.array(optimize[3:6])
    Ki = np.array(optimize[6:9])

    stick_index = int(d.time / 0.002)
    gyro_inputs = stickArray[stick_index]

    gyro_error = gyro_inputs - d.sensor('drone_gyro').data  

    if previous_error[0] == -1000:
        previous_error = gyro_error

    
        
    gyro_deriv = (gyro_error - previous_error) / m.opt.timestep
    previous_error[:] = gyro_error

    gyro_integ += m.opt.timestep * gyro_error
    
    u = Kp*gyro_error + Ki * gyro_integ + Kd * gyro_deriv

    thrust = 0.0
    
    final = actuator_arr @ np.concatenate((u, [thrust]))

    d.actuator('drone_fl').ctrl[0] = final[0]
    d.actuator('drone_fr').ctrl[0] = final[1]
    d.actuator('drone_br').ctrl[0] = final[2]
    d.actuator('drone_bl').ctrl[0] = final[3]



    
  
  except Exception as e:
    print(traceback.format_exc())   



stickArray_scaled =  stickArray * optimize[9:12]
def simulation_rollout(x, d, stickArray, prev_error, gyro_integ, accel_list, gyro_list, force_list, viewer = None):
  for i in range(stickArray.shape[0]):
    drone_control(x, m, d, stickArray, prev_error, gyro_integ)
    accel_list.append(np.copy(d.sensor('drone_acc').data))
    gyro_list.append(np.copy(d.sensor('drone_gyro').data))
    force_list.append(np.copy(d.sensor('drone_force').data))
    mujoco.mj_step(m, d)
    if viewer is not None:
      viewer.sync()
      time.sleep(0.002)


simulation_rollout(optimize, d, stickArray_scaled, prev_error, gyro_integ, accel_2_list, gyro_2_list, force_2_list, None)

accel_2_array = np.array(accel_2_list)
gyro_2_array = np.array(gyro_2_list)
force_2_array = np.array(force_2_list)
gyro_list_2 = np.hstack((timeArray, gyro_2_array)).tolist() # Load from data, first column time, 2,3,4th w_x, w_y, w_z

accel_i_array_2 = create_accelArray_rotated(gyro_list_2, accel_2_array)


plt.figure(1)
plt.plot(np.arange(force_2_array[:,0].size), force_2_array[:,0])
plt.plot(np.arange(accel_i_array[:,0].size), accel_i_array_2[:,0])
plt.show()
exit()


plt.figure(2)
#plt.plot(np.arange(force_2_array[:,0].size), force_2_array[:,1])
plt.plot(np.arange(accel_i_array[:,1].size), accel_i_array_2[:,1])


plt.figure(3)
#plt.plot(np.arange(force_2_array[:,0].size), force_2_array[:,2])
plt.plot(np.arange(accel_i_array[:,2].size), accel_i_array_2[:,2])

plt.show()


plt.figure(100)
plt.plot(np.arange(accel_i_array[:,0].size), accel_i_array[:,0])
plt.plot(np.arange(accel_i_array[:,1].size), accel_i_array[:,1])
plt.plot(np.arange(accel_i_array[:,2].size), accel_i_array[:,2])


plt.figure(200)
plt.plot(np.arange(accel_i_array[:,0].size), accel_i_array_2[:,0])
plt.plot(np.arange(accel_i_array[:,1].size), accel_i_array_2[:,1])
plt.plot(np.arange(accel_i_array[:,2].size), accel_i_array_2[:,2])
plt.show()
 
#TODO: rotate force to fixed frame then do 6 plots: first 3 are each force with each acceleration, then plot other three gyro values for other three