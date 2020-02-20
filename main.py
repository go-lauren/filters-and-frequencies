import skimage.io as skio
import skimage as sk
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from os import path
from filters import *
from stacks import *
from align_image_code import align_images
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
    # straigthen
    for f in ['groceries.png', 'facade.png', 'sidewalk.png', 'friends.png']:
        if not path.exists(file_path + 'straighten/' + f):
            im = skio.imread("images/" + f)
            gray_im = sk.img_as_float(im)
            gray_im = sk.color.rgb2gray(gray_im)
            straighten, _ = auto_straighten(gray_im)
            plt.imsave(file_path + "straighten/" + f, ndimage.rotate(im, straighten))
            print("angle: ", straighten, f)
    

def test():
    fig, axes = plt.subplots(nrows=1, ncols=3)
    im = skio.imread("images/smm.png")
    im = sk.img_as_float(im)
    im = sharpen(im, 1)
    plt.imsave("images/results/sharpen/smm.png", im)
    axes[0].imshow(im)
    plt.show()

if __name__ == "__main__":
    # test()
    main()