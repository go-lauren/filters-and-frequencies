from scipy import ndimage
import numpy as np
import cv2

def finite_difference(im):
    dx = ndimage.convolve1d(im, weights=[1, -1], axis=1)
    dy = ndimage.convolve1d(im, weights=[1, -1], axis=0)
    g = np.sqrt(dx**2 + dy**2)
    return dx, dy, g

def blur(im, sig):
    kernel = cv2.getGaussianKernel(6*sig -1, sig)
    kernel = np.outer(kernel, kernel)
    if len(im.shape) == 3:
        r = ndimage.convolve(im[:,:,0], kernel).reshape((im.shape[0], im.shape[1], 1))
        g = ndimage.convolve(im[:, :, 1], kernel).reshape((im.shape[0], im.shape[1], 1))
        b = ndimage.convolve(im[:, :, 2], kernel).reshape((im.shape[0], im.shape[1], 1))
        return np.concatenate((r,g, b), axis=2)
    im = ndimage.convolve(im, kernel)
    return im

def two_conv(im):
    kernel = cv2.getGaussianKernel(17, 3)
    kernel = np.outer(kernel, kernel)
    im = ndimage.convolve(im, kernel)
    return finite_difference(im)

def derivative_of_gaussian(im):
    kernel = cv2.getGaussianKernel(17, 3)
    kernel = np.outer(kernel, kernel)
    dx_filter = ndimage.convolve1d(kernel, weights=[1, -1], axis=1)
    dy_filter = ndimage.convolve1d(kernel, weights=[1, -1], axis=0)
    dx = ndimage.convolve(im, dx_filter)
    dy = ndimage.convolve(im, dy_filter)
    g = np.sqrt(dx**2 + dy**2)
    return dx, dy, g

def gradient_angles(im):
    dx, dy, g = derivative_of_gaussian(im)
    o = np.abs(np.arctan2(dy, dx))
    o = np.pi * np.ones(o.shape)*(o > np.pi/2) - o
    o = np.abs(o)
    o = np.where(g > 0.02, o, -1)
    return o

def auto_straighten(im):
    angle = 0
    curr_max = 0
    width = np.floor(0.4*im.shape[1]).astype(np.int)
    height = np.floor(0.4*im.shape[0]).astype(np.int)
    for theta in range(-30, 31):
        if theta == 0:
            continue
        o = gradient_angles(ndimage.rotate(im, theta, reshape=False))
        o = ndimage.rotate(o, -theta, reshape=False)
        o = o[height:o.shape[0]-height, width:o.shape[1]-width]
        temp = ((0.0 <= o) & (o <= 0.02)).sum() + ((np.pi/2-0.02 <= o) & (o <= np.pi/2)).sum() 
        # print(temp, (o != -1).sum())
        temp /= (o != -1).sum()
        print(theta, temp)
        if temp > curr_max:
            angle = theta
            curr_max = temp
    return angle, o

def sharpen(im, scalar):
    kernel = cv2.getGaussianKernel(17, 3)
    kernel = np.outer(kernel, kernel)*scalar*-1
    kernel[8, 8] += (1 + scalar)
    if len(im.shape) == 3:
        r = ndimage.convolve(im[:,:,0], kernel).reshape((im.shape[0], im.shape[1], 1))
        g = ndimage.convolve(im[:, :, 1], kernel).reshape((im.shape[0], im.shape[1], 1))
        b = ndimage.convolve(im[:, :, 2], kernel).reshape((im.shape[0], im.shape[1], 1))
        return np.concatenate((r,g, b), axis=2)
    return ndimage.convolve(im, kernel)
    #return im + (im - kernel * im)
    # return (1 + 1 - kernel)

def hybrid(im1, im2, sig1, sig2):
    if len(im1.shape) == 3:
        r_l, r_h, r_hyb = hybrid(im1[:, :, 0], im2[:, :, 0], sig1, sig2)
        g_l, g_h, g_hyb  = hybrid(im1[:, :, 1], im2[:, :, 1], sig1, sig2)
        b_l, b_h, b_hyb = hybrid(im1[:, :, 2], im2[:, :, 2], sig1, sig2)
        low_pass = np.concatenate((
            r_l.reshape((im1.shape[0], im2.shape[1], 1)),
            g_l.reshape((im1.shape[0], im2.shape[1], 1)),
            b_l.reshape((im1.shape[0], im2.shape[1], 1))
        ), axis = 2)
        high_pass = np.concatenate((
            r_h.reshape((im1.shape[0], im2.shape[1], 1)),
            g_h.reshape((im1.shape[0], im2.shape[1], 1)),
            b_h.reshape((im1.shape[0], im2.shape[1], 1))
        ), axis = 2)
        hyb = np.concatenate((
            r_hyb.reshape((im1.shape[0], im2.shape[1], 1)),
            g_hyb.reshape((im1.shape[0], im2.shape[1], 1)),
            b_hyb.reshape((im1.shape[0], im2.shape[1], 1))
        ), axis = 2)
        return low_pass, high_pass, hyb
    lp_k = cv2.getGaussianKernel(np.floor(sig1*6-1).astype(np.int), sig1)
    lp_k = np.outer(lp_k, lp_k)
    low_pass = ndimage.convolve(im1, lp_k)

    hp_k = cv2.getGaussianKernel(np.floor(sig2*6-1).astype(np.int), sig2)
    hp_k = np.outer(hp_k, hp_k)*-1
    hp_k[(hp_k.shape[0]-1)//2, (hp_k.shape[0]-1)//2] += 1
    high_pass = ndimage.convolve(im2, hp_k)

    h = high_pass + low_pass
    return low_pass, high_pass, h
