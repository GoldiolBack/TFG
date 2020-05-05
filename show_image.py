import numpy as np
import matplotlib.pyplot as plt


input = (np.loadtxt('val_input.csv', delimiter=','))
output = (np.loadtxt('val_output.csv', delimiter=','))
real = (np.loadtxt('val_real.csv', delimiter=','))

input = input.reshape((5, 32, 32, 6))
output = output.reshape((5, 64, 64, 6))
real = real.reshape((5, 64, 64, 6))


def show_im(a, b, c):
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    ax1.imshow(a[:, :, 2:5]/a[:, :, 2:5].max())
    ax2.imshow(b[:, :, 2:5]/b[:, :, 2:5].max())
    ax3.imshow(c[:, :, 2:5]/c[:, :, 2:5].max())
    ax4.imshow(a[:, :, 3:6] / a[:, :, 3:6].max())
    ax5.imshow(b[:, :, 3:6] / b[:, :, 3:6].max())
    ax6.imshow(c[:, :, 3:6] / c[:, :, 3:6].max())
    plt.show()


show_im(input[4], output[4], real[4])
