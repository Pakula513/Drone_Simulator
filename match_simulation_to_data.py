import mujoco
import mujoco.viewer as viewer
import numpy as np
import traceback
import time
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description = 'Optimize inputs so that the drone in the simulation matches a real flight')
parser.add_argument('--path', dest = 'fpath')
args = parser.parse_args()
filepath = args.fpath

if not filepath:
  filepath = "drone_test_1_BB.csv"






counter = 0
test_counter = 0

blackbox_df = pd.read_csv(filepath, skiprows = 141, 
                          usecols = ['loopIteration', 'time', 'rcCommand[0]', 'rcCommand[1]', 'rcCommand[2]', 'rcCommand[3]', 'gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]'])


pitchArray = np.array(blackbox_df.loc[:, 'rcCommand[1]'].to_numpy(), ndmin=2).T
rollArray = np.array(blackbox_df.loc[:, 'rcCommand[0]'].to_numpy(), ndmin=2).T
yawArray = np.array(blackbox_df.loc[:, 'rcCommand[2]'].to_numpy(), ndmin=2).T



stickArray_full = np.hstack((rollArray, pitchArray, yawArray))
stickArray = np.copy(stickArray_full)
stickArray = stickArray[0:(int(stickArray[:,0].size/4))]

stickArray = stickArray * (np.pi/180)

thrustArray_full = np.array(blackbox_df.loc[:, 'rcCommand[3]'].to_numpy(), ndmin=2).T
thrustArray = np.copy(thrustArray_full)
thrustArray = thrustArray[0:(int(thrustArray[:,0].size/4))]



gyro_x = np.array(blackbox_df.loc[:, 'gyroADC[0]'].to_numpy(), ndmin = 2).T
gyro_y = np.array(blackbox_df.loc[:, 'gyroADC[1]'].to_numpy(), ndmin = 2).T
gyro_z = np.array(blackbox_df.loc[:, 'gyroADC[2]'].to_numpy(), ndmin = 2).T


bodyrates_groundt_full = np.hstack((gyro_x, gyro_y, gyro_z))
bodyrates_groundt = np.copy(bodyrates_groundt_full)
bodyrates_groundt = bodyrates_groundt[0:(int(bodyrates_groundt[:,0].size/4))]

bodyrates_groundt = bodyrates_groundt * (np.pi/180)




actuator_arr = np.array([[ 1,-1, 1, 1],
                         [ 1, 1,-1, 1],
                         [-1, 1, 1, 1],
                         [-1,-1,-1, 1]])




##### Controller initialization #################

def drone_control(optimize, m, d, thrustArray, stickArray, previous_error, gyro_integ):

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




def simulation_rollout(x, d, thrustArray_scaled, stickArray_scaled, prev_error, gyro_integ, bodyrates_list, viewer = None):
  for i in range(bodyrates_groundt.shape[0]):
    drone_control(x, m, d, thrustArray_scaled, stickArray_scaled, prev_error, gyro_integ)
    bodyrates_list.append(np.copy(d.sensor('drone_gyro').data))
    mujoco.mj_step(m, d)
    if viewer is not None:
      viewer.sync()
      time.sleep(0.002)




def bodyrates_error(x, results=False, view=None):

  prev_error = np.array((-1000,-1000,-1000))

  stickArray_scaled = stickArray * x[9:12]
  thrustArray_scaled = thrustArray * 0 # x[6]

  gyro_integ = np.zeros((3,))

  d = mujoco.MjData(m)

  bodyrates_list = []
  
  if view:
    with mujoco.viewer.launch_passive(m,d) as viewer:
      simulation_rollout(x, d, thrustArray_scaled, stickArray_scaled, prev_error, gyro_integ, bodyrates_list, viewer)
  else:
    simulation_rollout(x, d, thrustArray_scaled, stickArray_scaled, prev_error, gyro_integ, bodyrates_list)

  bodyrates_array = np.array(bodyrates_list)

  ans = bodyrates_array.flatten() - bodyrates_groundt.flatten()

  if not results:
    return ans / np.sqrt(bodyrates_groundt.shape[0])
  else:
    return bodyrates_array    





################ Processing ################
if __name__ == '__main__':


  m = mujoco.MjModel.from_xml_path('hoopflyt_noground.xml')

  


  x0 = [0.01,0.01,0.01,
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001,
        1.0,1.0,1.0]
  x_low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  x_high = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
  bounds=(x_low, x_high)

  optimized_results = scipy.optimize.least_squares(lambda x: bodyrates_error(x), x0, bounds=bounds, verbose=2, ftol=1e-2)#, diff_step=1.0)  
  xstar = optimized_results.x


  print(x0)
  print(xstar)

  optimal_results = bodyrates_error(xstar, results=True, view=False)


  
  print("plotting...")

  plt.subplot(311)
  plt.plot(optimal_results[:, 1], label='pitch_measured')
  plt.plot(bodyrates_groundt[:, 1], label='pitch_groundtruth')
  plt.plot(stickArray[:, 1]*xstar[9+1], label='pitch_input')
  plt.grid()
  plt.legend()
  plt.subplot(312)
  plt.plot(optimal_results[:, 0], label='roll_measured')
  plt.plot(bodyrates_groundt[:, 0], label='roll_groundtruth')
  plt.plot(stickArray[:, 0]*xstar[9+0], label='roll_input')
  plt.grid()
  plt.legend()

  plt.subplot(313)
  plt.plot(optimal_results[:, 2], label='yaw_measured')
  plt.plot(bodyrates_groundt[:, 2], label='yaw_groundtruth')
  plt.plot(stickArray[:, 2]*xstar[9+2], label='yaw_input')
  plt.grid()
  plt.legend()

  plt.show()