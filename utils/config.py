import os

import torch

track = 'single-subject'     # single-subject or cross-subject
task = 'audio_only'          # audio_only or audio_video
sub_id = 'ABCDEFGH'
sub_num = len(sub_id)        # 8名被试
is_origin = False            # 针对 single-subject，是否使用 origin_data

if track == 'single-subject':
    if is_origin:
        # ./origin_data/audio_only/SA/train_eeg.npy
        data_dir = f'./origin_data/{task}'
        label_dir = f'../origin_data/label'
    else:
        # ./Data_for_SS/audio_only/SA/train_eeg.npy
        # ./Data_for_SS/audio_only/SA/train_label.npy
        data_dir = f'./Data_for_SS/{task}'
        label_dir = f'./Data_for_SS/{task}'
elif track == 'cross-subject':
    # ./origin_data/audio_only/SA.npy
    data_dir = f'./Data_for_CS/{task}'
    label_dir = f'../origin_data/label'

# k折交叉验证(train_valid)
k_fold = 5
# 超参数设置
batch_size = 64
# Adam优化器学习率等
lr = 0.001
b1 = 0.9
b2 = 0.999
# 早停法参数设置
epochs_stage1 = 1500
no_improvement_times = 200
epochs_stage2 = 600
model_name = 'base'     # base: 测试模型
results_dir = f'./results/{model_name}_{epochs_stage1}_{batch_size}'    # 结果日志记录
params_dir = f'./params/{model_name}_{epochs_stage1}_{batch_size}'      # 模型权重保存

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
