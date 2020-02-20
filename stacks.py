from scipy import ndimage
import numpy as np
import cv2

def stack(im, n):
    if len(im.shape) == 3:
        r_g, r_l = stack(im[:, :, 0], n)
        b_g, b_l = stack(im[:, :, 1], n)
        g_g, g_l = stack(im[:, :, 2], n)
        gauss_stack = np.zeros((im.shape[0], im.shape[1], 3, n+1))
        lap_stack = np.zeros((im.shape[0], im.shape[1], 3, n+1))
        for i in range(0, n+1):
            temp = r_g[:, :, i].reshape((im.shape[0], im.shape[1], 1, 1))
            gauss_stack[:, :, :, i] = np.concatenate((
                r_g[:, :, i].reshape((im.shape[0], im.shape[1], 1)),
                b_g[:, :, i].reshape((im.shape[0], im.shape[1], 1)),
                g_g[:, :, i].reshape((im.shape[0], im.shape[1], 1))
            ), axis=2)

            lap_stack[:, :, :, i] = np.concatenate(
                (r_l[:, :, i].reshape((im.shape[0], im.shape[1], 1)),
                b_l[:, :, i].reshape((im.shape[0], im.shape[1], 1)),
                b_g[:, :, i].reshape((im.shape[0], im.shape[1], 1))
            ), axis=2)
        return gauss_stack, lap_stack
    gauss_stack = np.zeros((im.shape[0], im.shape[1], n+1))
    lap_stack = np.zeros((im.shape[0], im.shape[1], n+1))
    gauss_stack[:, :, 0] = im
    for i in range(1, n+1):
        sig = 2**(i+1)
        ker = cv2.getGaussianKernel(min(6*sig-1, 100), sig)
        ker = np.outer(ker, ker)
        gauss_stack[:, :, i] = ndimage.convolve(im, ker)
        lap_stack[:, :, i-1] = gauss_stack[:, :, i-1] - gauss_stack[:, :, i]
    lap_stack[:, :, -1] = gauss_stack[:, :, -1]
    return gauss_stack, lap_stack

def spline(im1, im2, mask, n):
    _, l1 = stack(im1, n)
    _, l2 = stack(im2, n)
    g, _ = stack(mask, n)
    spline = np.zeros(g.shape)
    for i in range(0, n+1):
        spline[:, :, i] = g[:, :, i]*l1[:, :, i] + (np.ones(g.shape[0:2]) - g[:, :, i])*l2[:, :, i]
    return spline
