##==================================================
# Vu Hoang Minh, MAIA
# Lab 3 : Image Processing
##==================================================

# Import the library as show images, plot, etc.
import matplotlib.pyplot as plt
# Import the library as show images, plot, etc.
import matplotlib.pyplot as plt
# Import functionality for the color map
import matplotlib.cm as cm

# Import system specific parameters and function
import sys

## Other plotting libraries
# import seaborn as sns

# Import the library to mange the matrix and array
import numpy as np
import cv2

# Import the library to mange the ndimage
import scipy.ndimage
from scipy.misc import imresize

# Importing image processing toolbox
## Module to read, write,...
from skimage import io
## Module to convert the image on 8 bits
from skimage import img_as_ubyte
## Module to convert the image to float
from skimage import img_as_float
## Module for color conversion
from skimage import color
## Module image transform from skimage for resize
from skimage import transform
## Module misc from scipy for resize
from scipy import misc
## Module util from skimage
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from scipy.ndimage import convolve

# ===============================================================
# Resizing an image size using different interpolation functions
# ===============================================================
# Path = './images/'
# image_name = 'lena-grey.bmp'
# lena = io.imread(Path.__add__(image_name))
lena = io.imread('images/lena-grey.bmp')
vibot = color.rgb2grey(io.imread('images/vibot-color.jpg'))
# Get height and width of image
mlena, nlena = lena.shape
mvibot, nvibot = vibot.shape


#---------------------------------------------------------------
# Show original image
plt.figure()
io.imshow(lena)
plt.axis('off')
plt.suptitle('Lena')

#---------------------------------------------------------------
# Image Resize: Nearest Interpolation
scale = 2
nearest_lena = imresize(lena, (mlena * scale, nlena * scale), interp='nearest')
plt.figure()
io.imshow(nearest_lena)
plt.axis('off')
plt.suptitle('Nearest Interpolation with Scale=2')

#---------------------------------------------------------------
# Image Resize: Bilinear Interpolation
scale = 3
bilinear_lena = imresize(lena, (mlena * scale, nlena * scale), interp='bilinear')
plt.figure()
io.imshow(bilinear_lena)
plt.axis('off')
plt.suptitle('Bilinear Interpolation with Scale=3')

#---------------------------------------------------------------
# Image Resize: Bicubic Interpolation
scale = 4
bicubic_lena = imresize(lena, (mlena * scale, nlena * scale), interp='bicubic')
plt.figure()
io.imshow(bicubic_lena)
plt.axis('off')
plt.suptitle('Bicubic Interpolation with Scale=4')



# ===============================================================
# Blending algorithms declaration
# ===============================================================

# ---------------------------------------------------------------
# Resize image to be blended
def resizeImageToBlend(im1, im2):
    m_im1, n_im1 = im1.shape
    scaled_img2 = imresize(im2, (m_im1, n_im1), interp='bilinear')
    return scaled_img2

# ---------------------------------------------------------------
# Simple blending function
def blendSimple(im1, im2):
    # Resize image to be blended
    scaled_im2 = resizeImageToBlend(im1, im2)
    m_im1, n_im1 = im1.shape
    m_im2, n_im2 = scaled_im2.shape

    # Size of blended image
    mBlImage = m_im1
    nBlImage = (n_im1 + n_im2) / 2

    # Take values of Left-half of image 1 and Right-half of Image 2
    simpleBlendedImage = np.zeros([mBlImage, nBlImage])
    simpleBlendedImage[:, 0:n_im1 / 2 - 1] = im1[:, 0:n_im1 / 2 - 1]
    simpleBlendedImage[:, n_im1 / 2:nBlImage - 1] = scaled_im2[:, n_im1 / 2:nBlImage - 1]

    # Return simple blended image
    return simpleBlendedImage

# ---------------------------------------------------------------
# Alpha blending function
def blendAlpha(im1, im2):
    # Resize image to be blended
    scaled_im2 = resizeImageToBlend(im1, im2)
    m_im1, n_im1 = im1.shape
    m_im2, n_im2 = scaled_im2.shape

    simpleBlendedImage=blendSimple(im1, im2)

    # Get alpha blended image
    w = 200
    im1_Blend = im1[:, n_im1 / 2 - w / 2:n_im1 / 2 + w / 2]
    scaled_im2_Blend = scaled_im2[:, n_im2 / 2 - w / 2: n_im2 / 2 + w / 2]
    alphaBlendedImage = simpleBlendedImage

    for i in range(0, w):
        alpha_orange = -1.0 / w * i + 1
        alpha_apple = 1 - alpha_orange
        alphaBlendedImage[:, n_im1 / 2 - w / 2 + i] = alpha_orange * im1_Blend[:, i] + alpha_apple * scaled_im2_Blend[:, i]

    # Return simple blended image
    return alphaBlendedImage


# ---------------------------------------------------------------
# Pyramid blending function
def blendPyramid(im1, im2):
    # Resize image to be power of 2
    scaled_im1 = imresize(im1, (512, 512), interp='bilinear')
    scaled_im2 = imresize(im2, (512, 512), interp='bilinear')

    # generate Gaussian pyramid for A
    G = scaled_im1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = scaled_im2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((scaled_im1[:, :cols / 2], scaled_im2[:, cols / 2:]))
    return ls_

# ===============================================================
# Blending two images
# ===============================================================
orange = color.rgb2gray(io.imread('images/orange.jpeg'))
apple = color.rgb2gray(io.imread('images/apple.jpeg'))

# Simple blended
simpleBlendedImage=blendSimple(lena,apple)
plt.figure()
plt.axis('off')
plt.imshow(simpleBlendedImage, cmap = 'Greys_r')
plt.title('Simple blended image between orange and apple')

# Alpha blended
alphaBlendedImage=blendAlpha(lena,apple)
plt.figure()
plt.axis('off')
plt.imshow(alphaBlendedImage, cmap = 'Greys_r')
plt.title('Alpha blended image between orange and apple')

# Pyramid blended
pyramidBlendedImage=blendPyramid(lena,apple)
plt.figure()
plt.axis('off')
plt.imshow(pyramidBlendedImage, cmap = 'Greys_r')
plt.title('Pyramid blended image between orange and apple')

# ===============================================================
# Display results
# ===============================================================
plt.show()
