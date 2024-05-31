import os
import time

import utils.config as cfg
from utils.preprocessing import exponential_moving_standardize as standardization
from utils.train_valid_and_test import train_valid_model, test_model

import numpy as np
import torch
from sklearn.model_selection import KFold

# 设置随机种子，保证结果的可重复性
rand_seed = 2024
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(rand_seed)

# 验证集和训练集准确率
val_res = torch.zeros((cfg.sub_num, cfg.k_fold))
test_res = torch.zeros((cfg.sub_num, cfg.k_fold))
k_Fold = KFold(n_splits=cfg.k_fold, shuffle=True, random_state=rand_seed)

# 模型参数字典，键名必须和模型参数名一致。无关参数如导联个数可省略
model_params = {'kernel_size': 17}
results_dir_ = cfg.results_dir + '/'
for key, val in model_params.items():
    results_dir_ += f'_{key}{val}'
if not os.path.exists(results_dir_):
    os.makedirs(results_dir_)
results_write = open(f'{results_dir_}/results.txt', 'w')

start = time.time()

for sub, sub_id in enumerate('ABCDEFGH'):
    print(f'----------subject{sub_id}:----------')
    data_dir = f'./Data_for_SS/audio_only/S{sub_id}'
    eeg_train = np.load(data_dir + '/train_eeg.npy').transpose(0, 2, 1)  # (3872, 32, 128)
    label_train = np.load(data_dir + '/train_label.npy')  # (3872, 1)
    eeg_val = np.load(data_dir + '/val_eeg.npy').transpose(0, 2, 1)  # (480, 32, 128)
    label_val = np.load(data_dir + './val_label.npy')  # (480, 1)

    # standardization
    # Note that the preprocessing operations are trial-independent, so it is applicable for an online BCI.
    eeg_train, eeg_val = standardization(eeg_train), standardization(eeg_val)

    for fold, (train_ids, valid_ids) in enumerate(k_Fold.split(eeg_train)):
        val_res[sub, fold] = train_valid_model(eeg_train, label_train, train_ids, valid_ids, sub, fold + 1,
                                               model_params)
        test_res[sub, fold] = test_model(eeg_val, label_val, sub, fold + 1, model_params)
        results_write.write(
            f'sub{sub + 1}_fold{fold + 1}:\tval_acc:{val_res[sub, fold]:.4f}, test_acc:{test_res[sub, fold]:.4f}\n')
    break

duration = int(time.time() - start)  # 单位：秒
hour = duration // 3600
minute = duration % 3600 // 60
second = duration % 60

val_res_mean = torch.mean(val_res, dim=1)
results_write.write(f'***The average valid dataset accuracy is  {val_res_mean.mean()}\n{val_res_mean}')
test_res_mean = torch.mean(test_res, dim=1)
results_write.write(f'\n***The average test dataset accuracy is {test_res_mean.mean()}\n{test_res_mean}')
results_write.write(f'\nThe total train_valid_and_test duration is {hour}h-{minute}m-{second}s.\n')
results_write.close()
