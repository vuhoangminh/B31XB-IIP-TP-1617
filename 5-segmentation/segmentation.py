##==================================================
# Vu Hoang Minh, MAIA
# Lab 5 : Image Processing
##==================================================

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage.color import rgb2gray ,gray2rgb
from skimage.io import imread, imshow
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, binary_erosion, binary_dilation
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.draw import circle
from math import exp, expm1, floor
from PIL import Image



# ===============================================================
# Segmentation of toy ===========================================
# ===============================================================

# ---------------------------------------------------------------
# Read and rescale image
Path = './images/'
inputImage = 'coins.jpg'
coinsImage = imread(Path.__add__(inputImage))
coinsImage_ubyte = img_as_ubyte(coinsImage)
coinsImage_scaled = rescale(coinsImage_ubyte, 1 / 4)

# Show scaled image
imshow(coinsImage_scaled)  #Displaying the image
plt.title('Rescaled coins')
plt.axis('off')


# ---------------------------------------------------------------
# Convert image to gray scale
coinsImage_gray=rgb2gray(coinsImage_scaled)
plt.figure()
imshow(coinsImage_gray)
plt.title('Gray coins')
plt.axis('off')

# Threshold image using Otsu's thresholding
globalThreshold = threshold_otsu(coinsImage_gray)
otsuThreshold = coinsImage_gray <= globalThreshold

# Show thresholeded image
plt.figure()
imshow(otsuThreshold)
plt.title('Otsu thresholded coins')
plt.axis('off')


# ---------------------------------------------------------------
# Applying Different Morphological operations
disk1 = disk(2) #Setting disk size 4x4 for erosion

# Apply morphological operations
coinsImage_open = binary_opening(otsuThreshold, disk1, out = None)
coinsImage_close = binary_closing(otsuThreshold, disk1, out = None)

# Clean the image using new disk and openning operation of coinsImage_bin_close
disk2 = disk(6)
coinsImage_cleaned = binary_opening(coinsImage_close, disk2, out = None)

# Show images after some Morphological operations
plt.figure()
imshow(coinsImage_open)
plt.title('Morphological opening with disk(2)')
plt.axis('off')
plt.figure()
imshow(coinsImage_close)
plt.title('Morphological closing with disk(2)')
plt.figure()
plt.axis('off')
imshow(coinsImage_cleaned)
plt.title('Morphological closing with disk(2) followed by openning with disk(6)')
plt.axis('off')


# ---------------------------------------------------------------
# Find labelled image
labelledImage , numRegion  = label(coinsImage_cleaned, return_num = True, connectivity = 1)

# Show labelled image with number of found regions
plt.figure()
imshow(labelledImage)
plt.title('Labelled image with %d regions'%(numRegion))
plt.axis('off')


# Print number of found regions
print ("============================================")
print ("Part 1: ")
print ("Numer of found regions are : %d " % numRegion)

# Measure properties of labeled image regions
regionedImage = regionprops(labelledImage)


# ---------------------------------------------------------------
# Find and display the radius of each region
coinsImage_cleaned_gray = gray2rgb(img_as_ubyte(coinsImage_cleaned))
numLabel=0
for regionLabel in regionedImage:
    numLabel = numLabel+1
    radius = float(regionLabel["major_axis_length"] / 2) + 3  # +3 to make the circle cover the whole region
    [xCoordinate, yCoordinate] = circle(float(regionLabel["centroid"][0]),
                                        float(regionLabel["centroid"][1]),
                                        radius)
    # Display the radius of each region
    print("The radius of Region %d is %f" %(numLabel,radius))
    # Each region has different color code
    colorCode = floor(255/numRegion*(numLabel-1))
    coinsImage_cleaned_gray[xCoordinate, yCoordinate] = (colorCode, colorCode, colorCode)

# Draw the corresponding circles on the image
plt.figure()
imshow(coinsImage_cleaned_gray)
plt.title("Circles coins with different labels")
plt.axis('off')




# ===============================================================
# Segmentation of markers =======================================
# ===============================================================

# ---------------------------------------------------------------
# Read and display image
Path = './images/'  #Adding Path of the image
inputImage = 'objets4.jpg'  #Image name
markersImage = imread(Path.__add__(inputImage)) #Reading the image
plt.figure()
imshow(markersImage)  #Displaying the image
plt.title('Markers')
plt.axis('off')


# ---------------------------------------------------------------
# Convert image to gray scale
markersImage_ubyte = img_as_ubyte(markersImage)  #Image as Ubyte
markersImage_gray = rgb2gray(markersImage_ubyte)

plt.figure()
imshow(markersImage_gray)  #Displaying the image
plt.title('Gray markers')
plt.axis('off')

# Threshold image using Otsu's thresholding
globalThreshold = threshold_otsu(markersImage_gray) #applying threshiold_otsu function
otsuThreshold = markersImage_gray < globalThreshold  #Setting threshold
plt.figure()
imshow(otsuThreshold) #Showing the image
plt.title('Otsu thresholded markers')
plt.axis('off')


# ---------------------------------------------------------------
# Applying Different Morphological operations
disk1 = disk(8)

# Apply morphological operations to clean the image
markersImage_cleaned = binary_closing(otsuThreshold, disk1, out = None)

# Show clean image
plt.figure()
imshow(markersImage_cleaned)
plt.title('Morphological closing with disk(8)')
plt.axis('off')


# ---------------------------------------------------------------
# Using segmentation find the number of each object in the image.
# Note: the result above can be considered a cleaned image. However to find the number
#       each object (marker or glue) we have to use dilation operation, so that different
#       part of one object can be connected into one
disk2 = disk(6)
markersImage_dilated = binary_dilation(otsuThreshold, disk2, out = None)
plt.figure()
imshow(markersImage_dilated)
plt.title('Morphological closing with disk(8) followed by dilation with disk(6)')
plt.axis('off')


# ---------------------------------------------------------------
# Find labelled image
labelledImage, numRegion = label(markersImage_dilated, return_num=True)
regionedImage = regionprops(labelledImage)

# Show labelled image with number of found regions
plt.figure()
imshow(labelledImage)
plt.title('Labelled image with %d regions'%(numRegion))
plt.axis('off')

# Print number of found regions
print ("============================================")
print ("Part 2: ")
print ("Numer of found regions are : %d " % numRegion)

# Find the number of glue and marker in the image
# Note: I realized that the length of marker is bigger than glue's
#       and the offset I found is 100
#       Using major_axis_length, property of regionprops, to compare to
#       100, I can find the number of each object
numMarkers = 0
numGlues = 0
numRow, numColumn = markersImage_gray.shape
colorImage = np.zeros(shape=(numRow, numColumn))
for regionLabel in regionedImage:
    radius = float(regionLabel["major_axis_length"] / 2)
    if (radius>100):
        numMarkers = numMarkers + 1
        for point in regionLabel["coords"]:
            colorImage[point[0],point[1]] = 0.6		# paint color 
    else:
        numGlues = numGlues + 1
        for point in regionLabel["coords"]:
            colorImage[point[0],point[1]] = 1		# paint color 

print("The number of markers are : %d" % numMarkers)
print("The number of gum are : %d" % numGlues)

plt.figure()
imshow(colorImage)
plt.title("Different object with different color")
plt.axis('off')




# ===============================================================
# Display results
# ===============================================================
plt.show()


