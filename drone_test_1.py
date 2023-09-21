import numpy as np
import pandas as pd
from scipy import signal


# rcCommand 0 is roll, 1 is pitch, 2 is yaw
blackbox_df = pd.read_csv(r'C:\\Users\\Akula\\Documents\\Summer2023_Stuff\\06_23_2023_Data\\drone_test_1_BB.csv', skiprows = lambda x: x in range(141)
                 ,usecols = ['rcCommand[0]', 'rcCommand[1]', 'rcCommand[2]', 'gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]'])



pitchArray = np.array(blackbox_df.loc[:, 'rcCommand[1]'].to_numpy(), ndmin = 2).T
rollArray = np.array(blackbox_df.loc[:, 'rcCommand[0]'].to_numpy(), ndmin = 2).T
yawArray = np.array(blackbox_df.loc[:, 'rcCommand[2]'].to_numpy(), ndmin = 2).T
stickArray = np.hstack((pitchArray, rollArray, yawArray))



gyro_x = np.array(blackbox_df.loc[:, 'gyroADC[0]'].to_numpy(), ndmin = 2).T
gyro_y = np.array(blackbox_df.loc[:, 'gyroADC[1]'].to_numpy(), ndmin = 2).T
gyro_z = np.array(blackbox_df.loc[:, 'gyroADC[2]'].to_numpy(), ndmin = 2).T


sos = signal.butter(7, 100, fs = 500, output = 'sos')
filtered = signal.sosfilt(sos, gyro_x)
filtered2 = signal.sosfilt(sos, gyro_y)
filtered3 = signal.sosfilt(sos, gyro_z)


filtered_gyroArray = np.hstack((filtered, filtered2, filtered3)) 




np.save('Documents/Summer2023_Stuff/data/06_23_2023_x_Array', stickArray)
np.save('Documents/Summer2023_Stuff/data/06_23_2023_y_Array', filtered_gyroArray)

