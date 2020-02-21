import skimage.io as skio
import skimage as sk
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from os import path
from filters import *
from stacks import *
from align_image_code import *
import os

def main():
    file_path = "images/results/"
    # edge detection
    if not path.exists(file_path + "cameraman/fd.png"):
        im = skio.imread("images/cameraman.png")
        im = sk.img_as_float(im)
        im = sk.color.rgb2gray(im)
        dx, dy, fd_im = finite_difference(im)
        plt.imsave(file_path + "cameraman/fd_x.png", dx, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/fd_y.png", dy, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/fd.png", fd_im, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/fd_bin.png", np.where(fd_im > .22, 1, 0), cmap=plt.cm.gray)
    if not path.exists(file_path + "cameraman/2conv.png"):
        im = skio.imread("images/cameraman.png")
        im = sk.img_as_float(im)
        im = sk.color.rgb2gray(im)
        dx, dy, dconv_im = two_conv(im)
        plt.imsave(file_path + "cameraman/2conv_x.png", dx, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/2conv_y.png", dy, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/2conv.png", dconv_im, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/2conv_bin.png", np.where(dconv_im > .02, 1, 0), cmap=plt.cm.gray)
    if not path.exists(file_path + "cameraman/dog.png"):
        im = skio.imread("images/cameraman.png")
        im = sk.img_as_float(im)
        im = sk.color.rgb2gray(im)
        dx, dy, dog_im = derivative_of_gaussian(im)
        plt.imsave(file_path + "cameraman/dog_x.png", dx, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/dog_y.png", dy, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/dog.png", dog_im, cmap=plt.cm.gray)
        plt.imsave(file_path + "cameraman/dog_bin.png", np.where(dog_im > .02, 1, 0), cmap=plt.cm.gray)
    # sharpen
    for f in ['taj.png', 'smm.png']:
        if not path.exists(file_path + "sharpen/" + f):
            im = skio.imread("images/" + f)
            im = sk.img_as_float(im)
            im = sharpen(taj, 0.5)
            plt.imsave(file_path + "sharpen/" + f, taj)
    if not path.exists(file_path + "sharpen/dutch_blur.png"):
        dutch = skio.imread("images/dutchflat.png")
        dutch = sk.img_as_float(dutch)
        dutch_blur = blur(dutch, 3)
        dutch = sharpen(dutch_blur, 1)
        plt.imsave(file_path + "sharpen/dutch_blur.png", dutch_blur)
        plt.imsave(file_path + "sharpen/dutch_sharp.png", dutch)
    # straighten
    for f in ['groceries.png', 'facade.png', 'sidewalk.png', 'dog.png', 'pisa.png']:
        if not path.exists(file_path + 'straighten/' + f):
            im = skio.imread("images/" + f)
            gray_im = sk.img_as_float(im)
            gray_im = sk.color.rgb2gray(gray_im)
            straighten, _ = auto_straighten(gray_im)
            plt.imsave(file_path + "straighten/" + f, ndimage.rotate(im, straighten))
            print("angle: ", straighten, f)
    # stack
    for f in ['lincoln.png', 'flower.png', 'monalisa.png', 'hyb_me.png']:
        if not path.exists(file_path + 'stacks/' + "gs0" + f):
            im = skio.imread("images/" + f)
            im = sk.img_as_float(im)
            gs, ls = stack(im, 5)
            for i in range(0, 6):
                if i==5:
                    continue
                for j in range(0, 3):
                    channel = ls[:, :, j, i]
                    channel = channel - np.min(channel.ravel())
                    channel = channel/np.max(channel.ravel())
                    ls[:, :, j, i] = channel
            for i in range(0, 6):
                plt.imsave("images/results/stacks/gs" + str(i) + f, gs[:, :, :, i])
                plt.imsave("images/results/stacks/ls" + str(i) + f, ls[:, :, :, i])
    #hybrid
    if not path.exists(file_path + 'hybrid/' + 'hyb_noggo.png'):
        im1 = skio.imread("images/nina.png")
        im2 = skio.imread("images/doggo.png")
        im1 = sk.img_as_float(im1)
        im2 = sk.img_as_float(im2)
        im1 = sk.color.rgb2gray(im1)
        im2 = sk.color.rgb2gray(im2)
        lp, hp, hyb = hybrid(im1, im2, 10, 15)
        plt.imsave("images/results/hybrid/lp_nina.png", lp, cmap=plt.cm.gray)
        plt.imsave('images/results/hybrid/hp_doggo.png', hp, cmap=plt.cm.gray)
        plt.imsave('images/results/hybrid/hyb_noggo.png', hyb, cmap=plt.cm.gray)       
    if not path.exists(file_path + "hybrid/hyb_me.png"):
        im2 = skio.imread("images/neutral.png")
        im1 = skio.imread("images/angry.png")
        im1 = sk.img_as_float(im1)
        im2 = sk.img_as_float(im2)
    
        lp, hp, hyb = hybrid(im1, im2, 5, 10)
        plt.imsave("images/results/hybrid/lp_me.png", lp)
        plt.imsave('images/results/hybrid/hp_me.png', hp)
        plt.imsave('images/results/hybrid/hyb_me.png', hyb)

        im1 = sk.color.rgb2gray(im1)
        im2 = sk.color.rgb2gray(im2)
        lp, hp, hyb = hybrid(im1, im2, 5, 10)
        plt.imsave("images/results/hybrid/gray_lp_me.png", lp, cmap=plt.cm.gray)
        plt.imsave('images/results/hybrid/gray_hp_me.png', hp, cmap=plt.cm.gray)
        plt.imsave('images/results/hybrid/gray_hyb_me.png', hyb, cmap=plt.cm.gray)
    if not path.exists(file_path + "hybrid/nut.png"):
        im1 = skio.imread("images/derek_aligned.jpg")
        im2 = skio.imread("images/nutmeg_aligned.jpg")
        im1 = sk.img_as_float(im1)
        im2 = sk.img_as_float(im2)
        im1 = sk.color.rgb2gray(im1)
        im2 = sk.color.rgb2gray(im2)
        lp, hp, hyb = hybrid(im1, im2, 5, 15)
        plt.imsave("images/results/hybrid/lp_derek.png", lp, cmap=plt.cm.gray)
        plt.imsave('images/results/hybrid/lp_nutmeg.png', hp, cmap=plt.cm.gray)
        plt.imsave('images/results/hybrid/nut.png', hyb, cmap=plt.cm.gray)
    if not path.exists(file_path + "blend/orapple.png"):
        im2 = skio.imread("images/apple.jpeg")
        im1 = skio.imread("images/orange.jpeg")
        im1 = sk.img_as_float(im1)
        im2 = sk.img_as_float(im2)
        l = im1.shape[1] // 2
        mask = np.ones((im1.shape[0], im1.shape[1]))
        mask[:, 0:l] = np.zeros((im1.shape[0], l))
        sp = spline(im1, im2, mask, 5)
        r = np.sum(sp[:, :, :, 0], axis=2)
        g = np.sum(sp[:, :, :, 1], axis=2)
        b = np.sum(sp[:, :, :, 2], axis=2)
        final = np.concatenate(
                    (r.reshape((im1.shape[0], im1.shape[1], 1)),
                    g.reshape((im1.shape[0], im1.shape[1], 1)),
                    b.reshape((im1.shape[0], im1.shape[1], 1))
                ), axis=2)
        plt.imsave("images/results/blend/orapple.png", final)
    if not path.exists(file_path + "blend/handeye.png"):
        im1 = plt.imread("images/eye.jpg")
        im2 = plt.imread("images/hand.png")
        im1 = sk.img_as_float(im1)
        im2 = sk.img_as_float(im2)
        l = im1.shape[1] // 2
        mask = skio.imread("images/eyemask.png")
        mask = sk.img_as_float(mask)
        mask = sk.color.rgb2gray(mask)
        sp = spline(im1, im2, mask, 5)
        r = np.sum(sp[:, :, :, 0], axis=2)
        g = np.sum(sp[:, :, :, 1], axis=2)
        b = np.sum(sp[:, :, :, 2], axis=2)
        final = np.concatenate(
                    (r.reshape((im1.shape[0], im1.shape[1], 1)),
                    g.reshape((im1.shape[0], im1.shape[1], 1)),
                    b.reshape((im1.shape[0], im1.shape[1], 1))
                ), axis=2)
        plt.imsave("images/results/blend/handeye.png", final)
def test():
    return
      
if __name__ == "__main__":
    test()
    main()