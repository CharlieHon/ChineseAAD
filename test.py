import time

import numpy as np

import config as cfg

print(cfg.data_dir)

dataA_dir = cfg.data_dir + '/SA'    # ./Data_for_SS/audio_only/SA/train_eeg/label.npy
train_eeg = np.load(dataA_dir + '/train_eeg.npy')
val_eeg = np.load(dataA_dir + '/val_eeg.npy')

if cfg.is_origin:
    train_label = np.load(cfg.label_dir + '/SA.npy')
    val_label = np.load(cfg.label_dir + '/SA.npy')
else:
    train_label = np.load(dataA_dir + '/train_label.npy')
    val_label = np.load(dataA_dir + '/val_label.npy')

# val_eeg = np.load(dataA_dir + '/val_eeg.npy')
# val_label = np.load(dataA_dir + '/val_label.npy')

print(train_eeg.shape)      # (3872, 128, 32)       # (16, 136800, 32)
print(train_label.shape)    # (3872, 1)             # (16, 1)
print(val_eeg.shape)        # (480, 128, 32)
print(val_label.shape)      # (480, 1)

# for i in 'ABCDEFGH':
#     print(f'S{i}')

# 无关测试
# duration = 165845
# hour = duration // 3600
# minute = duration % 3600 // 60
# second = duration % 60
# print(f'{duration}s = {hour}h-{minute}m-{second}s')
