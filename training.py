from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

(X, Y), (X_val, Y_val), axes = load_training_data('data/my_training_data.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

plt.figure(figsize=(12, 5))
plot_some(X_val[:5], Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');

config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=10)
print(config)
vars(config)

Config(axes='ZYXC', n_channel_in=1, n_channel_out=1, n_dim=3, probabilistic=False,
       train_batch_size=16, train_checkpoint='weights_best.h5', train_checkpoint_epoch='weights_now.h5',
       train_checkpoint_last='weights_last.h5', train_epochs=100, train_learning_rate=0.0004, train_loss='mae',
       train_reduce_lr={'factor': 0.5, 'patience': 10, 'min_delta': 0}, train_steps_per_epoch=10,
       train_tensorboard=True, unet_input_shape=(None, None, None, 1), unet_kern_size=3, unet_last_activation='linear',
       unet_n_depth=2, unet_n_first=32, unet_residual=True)

model = CARE(config, 'my_model', basedir='models')

history = model.train(X, Y, validation_data=(X_val, Y_val))

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
