from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches

download_and_extract_zip_file(
    url='/Users/sereenabhanji/Downloads/Condition_1_strainUM679_40xObjective.zip',
    targetdir='data',
)

raw_data = RawData.from_folder(
    basepath='data/tribolium/train',
    source_dirs=['low'],
    target_dir='GT',
    axes='ZYX',
)

X, Y, XY_axes = create_patches(
    raw_data=raw_data,
    patch_size=(16, 64, 64),
    n_patches_per_image=1024,
    save_file='data/my_training_data.npz',
)
assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

for i in range(2):
    plt.figure(figsize=(16, 4))
    sl = slice(8 * i, 8 * (i + 1)), 0
    plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
    plt.show()
