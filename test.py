import numpy as np
import matplotlib.pyplot as plt

from utils import config as cfg
from utils.preprocessing import exponential_moving_standardize as standardize

print(cfg.data_dir)

dataA_dir = cfg.data_dir + '/SA'  # ./Data_for_SS/audio_only/SA/train_eeg/label.npy
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

print(train_eeg.shape)  # (3872, 128, 32)       # (16, 136800, 32)
print(train_label.shape)  # (3872, 1)             # (16, 1)
print(val_eeg.shape)  # (480, 128, 32)
print(val_label.shape)  # (480, 1)

# 查看数据是否已经归一化(Data_for_SS)[是]
# eeg_avg = np.mean(train_eeg[0] * 1e6, axis=0)
# print(eeg_avg)
# eeg_std = np.std(train_eeg[0] * 1e6, axis=0)
# print(eeg_std)

# 测试，是这么计算均值和标准差的[是]
# a = np.arange(12).reshape(2, 2, 3)
# print(a)
# a_mean = np.mean(a, axis=1)
# print(a_mean)

# The exponential moving standardization(指数移动归一化)[OK]
# channel = 0
# trial = train_eeg[0].T
# trial_std = standardize(trial, 0.001)
# # trial_std_ = standardize(train_eeg[:5].transpose(0, 2, 1), 0.001)
# plt.plot(trial[channel] * 1e4, label='origin', color='green')
# # plt.plot(trial_std_[0][channel], label='n_samples', color='green')    # 效果一样
# plt.plot(trial_std[channel], label='standardize', color='blue')
# plt.legend()
# plt.show()

model_params = {'channel_num': 32, 'samples': 128, 'conv_num': 32}
# model_params/{'channel_num': 32, 'samples': 128, 'conv_num': 32}
print(f'model_params/{model_params}')

# print([f'_{key}{val}' for key, val in model_params.items()])

params_dir = cfg.params_dir + '/'
for key, val in model_params.items():
    params_dir += f'_{key}{val}'
print(params_dir)
