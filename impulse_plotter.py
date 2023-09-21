import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg
from scipy.signal import lfilter

x = np.load('Documents/Summer2023_Stuff/data/06_23_2023_x_Array.npy')
y = np.load('Documents/Summer2023_Stuff/data/06_23_2023_y_Array.npy')

# set both to 0 in order to not cut anything from graph
start_t = 0
end_t  = 0

if not (start_t == 0 and end_t == 0):
    x = np.delete(x, np.s_[int(start_t*1000):int(end_t*1000)], 0)
    y = np.delete(y, np.s_[int(start_t*1000):int(end_t*1000)], 0)


K = 800
N = x.shape[0]


def compute_A_h_conv(x, y):
    A = np.zeros((N-K+1, K+1))
    A = linalg.convolution_matrix(x, K, mode = "valid")
    h = np.linalg.lstsq(A, y, rcond = None)[0]
    conv = signal.correlate(x, h)

    return A, h, conv



##### Roll to gyro[1] impulse
y_2 = y[K-1:, 1]
A_2, h_2, conv_2 = compute_A_h_conv(x[:, 0], y[K-1:, 1])


##### Pitch to gyro [0] impulse
y_1 = y[K-1:,0]
A_1, h_1, conv_1 = compute_A_h_conv(x[:, 1], y[K-1:,0])


##### Yaw to gyro[2] impulse
y_3 = y[20000+K-1:30000,2]
A_3, h_3, conv_3 = compute_A_h_conv(x[20000:30000,2], y[20000+K-1:30000,2])





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
