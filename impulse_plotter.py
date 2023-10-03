import numpy as np
import pandas as pd
from scipy import signal
from scipy import linalg
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import argparse


def load_filter_bf_datafile(signal_order, signal_crit_freq, signal_sampling_freq, signal_output):

    # rcCommand 0 is roll, 1 is pitch, 2 is yaw
    blackbox_df = pd.read_csv(filepath, skiprows = 141, 
                            usecols = ['rcCommand[0]', 'rcCommand[1]', 'rcCommand[2]', 'gyroADC[0]', 'gyroADC[1]', 'gyroADC[2]'])
    

    pitchArray = np.array(blackbox_df.loc[:, 'rcCommand[1]'].to_numpy(), ndmin = 2).T
    rollArray = np.array(blackbox_df.loc[:, 'rcCommand[0]'].to_numpy(), ndmin = 2).T
    yawArray = np.array(blackbox_df.loc[:, 'rcCommand[2]'].to_numpy(), ndmin = 2).T
    stickArray = np.hstack((pitchArray, rollArray, yawArray))



    gyro_x = np.array(blackbox_df.loc[:, 'gyroADC[0]'].to_numpy(), ndmin = 2).T
    gyro_y = np.array(blackbox_df.loc[:, 'gyroADC[1]'].to_numpy(), ndmin = 2).T
    gyro_z = np.array(blackbox_df.loc[:, 'gyroADC[2]'].to_numpy(), ndmin = 2).T


    sos = signal.butter(signal_order, signal_crit_freq, fs = signal_sampling_freq, output = signal_output)
    filtered = signal.sosfilt(sos, gyro_x)
    filtered2 = signal.sosfilt(sos, gyro_y)
    filtered3 = signal.sosfilt(sos, gyro_z)


    filtered_gyroArray = np.hstack((filtered, filtered2, filtered3))

    return stickArray, filtered_gyroArray


def fit_impulse(x, y):
    A = np.zeros((N-K+1, K+1))
    A = linalg.convolution_matrix(x, K, mode = "valid")
    h = np.linalg.lstsq(A, y, rcond = None)[0]
    conv = signal.correlate(x, h)

    return A, h, conv



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Optimize inputs so that the drone in the simulation matches a real flight')
    parser.add_argument('--path', dest = 'fpath')
    args = parser.parse_args()
    filepath = args.fpath

    if not filepath:
        filepath = "drone_test_1_BB.csv"


    x, y = load_filter_bf_datafile(7, 100, 500, 'sos')


    # set both to 0 in order to not cut anything from graph
    start_t = 0
    end_t  = 0

    if not (start_t == 0 and end_t == 0):
        x = np.delete(x, np.s_[int(start_t*1000):int(end_t*1000)], 0)
        y = np.delete(y, np.s_[int(start_t*1000):int(end_t*1000)], 0)


    K = 800
    N = x.shape[0]





    ##### Roll to gyro[1] impulse
    y_2 = y[K-1:, 1]
    A_2, h_2, conv_2 = fit_impulse(x[:, 0], y[K-1:, 1])


    ##### Pitch to gyro [0] impulse
    y_1 = y[K-1:,0]
    A_1, h_1, conv_1 = fit_impulse(x[:, 1], y[K-1:,0])


    ##### Yaw to gyro[2] impulse
    y_3 = y[20000+K-1:30000,2]
    A_3, h_3, conv_3 = fit_impulse(x[20000:30000,2], y[20000+K-1:30000,2])





    ################ Plotting Impulses ##########################

    temp = 0.001 * np.arange(y_2.size)
    temp2 = 0.001 * np.arange(h_2.size)
    temp3 = 0.001 * np.arange((A_3 @ h_3).size)


    fig, axs = plt.subplots(3,2)
    fig.suptitle("Stick inputs and Gyro")



    axs[0,0].plot(temp, A_1 @ h_1, label = "A*h1")
    axs[0,0].plot(temp, y_1, label = "y_pitch")
    axs[0,1].plot(temp2, h_1, label = "impulse1")


    axs[1,0].plot(temp, A_2 @ h_2, label = "A*h2")
    axs[1,0].plot(temp, y_2, label = "y_roll")
    axs[1,1].plot(temp2, h_2, label = "impulse2")

    axs[2,0].plot(temp3, A_3 @ h_3, label = "A*h3")
    axs[2,0].plot(temp3, y_3, label = "y_yaw")
    axs[2,1].plot(temp2, h_3, label = "impulse3")

    plt.show()




