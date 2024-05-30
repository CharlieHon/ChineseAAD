import os

track = 'single-subject'     # single-subject or cross-subject
task = 'audio_only'          # audio_only or audio_video
sub_num = 8                  # 8名被试
sub_id = 'ABCDEFGH'
is_origin = False            # 针对 single-subject，是否使用 origin_data

if track == 'single-subject':
    if is_origin:
        # ./origin_data/audio_only/SA/train_eeg.npy
        data_dir = f'./origin_data/{task}'
        label_dir = f'./origin_data/label'
    else:
        # ./Data_for_SS/audio_only/SA/train_eeg.npy
        # ./Data_for_SS/audio_only/SA/train_label.npy
        data_dir = f'./Data_for_SS/{task}'
        label_dir = f'./Data_for_SS/{task}'
elif track == 'cross-subject':
    # ./origin_data/audio_only/SA.npy
    data_dir = f'./Data_for_CS/{task}'
    label_dir = f'./origin_data/label'
