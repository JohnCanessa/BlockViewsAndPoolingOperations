# **** ****
import numpy as np                              # numpy is primary library for numeric array (and matrix) processing
from matplotlib import pyplot as plt            # pyplot is sub-library of matplotlib, pyplot is for plotting

# **** ****
import skimage.io                               # skimage is scikit-image library for image processing
from skimage import color                       # skimage.color is sub-library for converting color spaces
from skimage.util import view_as_blocks         # skimage.util is sub-library for various generic utilities (like view_as_blocks)


# **** read three_dogs image ****
three_dogs = skimage.io.imread(fname='./images/pexels-3-dogs.jpg')

# **** plot three_dogs RGB image ****
plt.imshow( three_dogs,
            interpolation='nearest')            # plot image, set interpolation to nearest
plt.title('three_dogs - RGB')                   # set image title
plt.show()                                      # show image


# **** convert three_dogs to grayscale ****
three_dogs = color.rgb2gray(three_dogs)

# **** plot three_dogs grayscale image ****
plt.imshow( three_dogs, 
            cmap='gray')                        # plot image, set colormap to gray
plt.title('three_dogs - grayscale')             # set image title
plt.show()                                      # show image


# **** display shape of three_dogs (2D grayscale) ****
print(f'three_dogs.shape: {three_dogs.shape}')

# **** assign block spape 4 x 4 ****
block_shape = (4, 4)

# **** view three_dogs as blocks ****
three_dogs_blocks = view_as_blocks( three_dogs,
                                    block_shape=block_shape)

# **** display shape of three_dogs_blocks (H/4, W/4, 4, 4) ****
print(f'three_dogs_blocks.shape: {three_dogs_blocks.shape}')


# **** reshape three_dogs_blocks) ****
flattened_blocks = three_dogs_blocks.reshape(   three_dogs_blocks.shape[0],
                                                three_dogs_blocks.shape[1],
                                                -1)

# **** print shape of three_dogs_blocks ****
print(f'shape of the blocks image: {three_dogs_blocks.shape}')

# **** print shape of flattened image ****
print(f'shape of the flattened image: {flattened_blocks.shape}')


# **** mean-pooling: find the mean for each block ****
mean_blocks = np.mean(flattened_blocks, axis=2)

# **** plot mean_blocks ****
plt.imshow( mean_blocks,
            interpolation='nearest',            # plot image, set interpolation to nearest
            cmap='gray')                        # plot image, set colormap to gray
plt.title('mean_blocks - grayscale')            # set image title
plt.show()                                      # show image


# **** max-pooling: find the max for each block 
#      max_pooling is used to find the most prominent feature in each block ****

max_blocks = np.max(flattened_blocks, axis=2)

# **** plot max_blocks ****
plt.imshow( max_blocks,
            interpolation='nearest',            # plot image, set interpolation to nearest
            cmap='gray')                        # plot image, set colormap to gray
plt.title('max_blocks - grayscale')             # set image title
plt.show()                                      # show image


# **** median-pooling: find the median for each block ****
median_blocks = np.median(flattened_blocks, axis=2)

# **** plot median_blocks ****
plt.imshow( median_blocks,
            interpolation='nearest',            # plot image, set interpolation to nearest
            cmap='gray')                        # plot image, set colormap to gray
plt.title('median_blocks - grayscale')          # set image title
plt.show()                                      # show image
