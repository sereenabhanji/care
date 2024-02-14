from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread

from csbdeep.utils import plot_some
from csbdeep.data import RawData, create_patches

'''

y = imread('C:/Users/GerholdLab/Sereena/Condition_3_strainUM793_40xObjective/High_Res/2024_01_18_um793_1001-2.tif')
x = imread('C:/Users/GerholdLab/Sereena/Condition_3_strainUM793_40xObjective/Low_Res/2024_01_18_um793_1001-2.tif')
print('image size =', x.shape)

plt.figure(figsize=(16,10))
plot_some(np.stack([x,y]),
          title_list=[['low (maximum projection)','GT (maximum projection)']], 
          pmin=2,pmax=99.8);

'''

raw_data = RawData.from_folder(
    basepath='C:/Users/GerholdLab/Sereena/Condition_3_strainUM793_40xObjective',
    source_dirs=['low_Res'],
    target_dir='High_Res',
    axes='ZYX',
)

X, Y, XY_axes = create_patches(
    raw_data=raw_data,
    patch_size=(16, 64, 64),
    n_patches_per_image=20,
    save_file='C:/Users/GerholdLab/Sereena/Condition_3_strainUM793_40xObjective/data/my_training_data.npz',
)
assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

for i in range(2):
    plt.figure(figsize=(16, 4))
    sl = slice(8 * i, 8 * (i + 1)), 0
    plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
    plt.show()
