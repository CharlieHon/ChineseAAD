import os

import numpy as np
import tqdm
import torch
from torch import nn

import utils.config as cfg
from torch.utils.data import DataLoader
from utils.CustomDataset import MyDataset

from modules.ConvBase import ConvNet as Model


def train_valid_model(eeg, label, train_ids, valid_ids, sub, fold, model_params=None):
    print(f'---sub:{sub + 1}-fold:{fold}---')
    results_dir_ = cfg.results_dir + '/'
    for key, val in model_params.items():
        results_dir_ += f'_{key}{val}'
    results_dir_ += f'/sub{sub + 1}'
    if not os.path.exists(results_dir_):
        os.makedirs(results_dir_)
    subject_write = open(results_dir_ + f'/fold{fold}.txt', 'w')
    subject_write.write(f'---sub:{sub + 1}-fold:{fold}---\n')
    subject_write.write(f'---The early stopping stage 1.---\n')
    x_train, x_valid, y_train, y_valid = eeg[train_ids], eeg[valid_ids], label[train_ids], label[valid_ids]

    model = Model(**model_params).to(cfg.device)

    train_dataset = MyDataset(x_train, y_train)
    valid_dataset = MyDataset(x_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    # TODO: 修改为比较验证集准确率而非损失值
    valid_acc_max = -1.  # 验证集最高准确率
    valid_acc_max_loss = np.inf  # 验证集最高准确率下的损失值
    valid_acc_improvements = 0  # 验证集准确率连续未提高的次数

    # ---------- train adn valid ----------
    for epoch in range(cfg.epochs_stage1):
        # 正确分类数、训练样本总数、训练集损失值
        train_correct_num, train_num, train_loss = 0, 0, 0.
        i = 0
        # ---------- train ----------
        model.train()
        for step, (data, labels) in enumerate(tqdm.tqdm(train_loader, position=0, leave=False), start=1):
            data = data.to(cfg.device)
            labels = labels.to(cfg.device)

            pred = model(data)
            loss = criterion(pred, labels)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predictions = pred.max(1)
            train_correct_num += (predictions == labels).sum()
            train_num += predictions.size(0)
            i = step
        # 训练集准确率及损失，要保存的是在验证集得到最小损失时的train_acc, train_loss
        train_acc = float(train_correct_num) / float(train_num)
        train_loss /= i

        valid_correct_num, valid_num, valid_loss = 0, 0, 0.
        # ---------- valid ----------
        model.eval()
        for step, (data, labels) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=False), start=1):
            with torch.no_grad():
                data = data.to(cfg.device)
                labels = labels.to(cfg.device)

                pred = model(data)
                loss = criterion(pred, labels)
                valid_loss += loss

                _, predictions = pred.max(1)
                valid_correct_num += (predictions == labels).sum()
                valid_num += predictions.size(0)
            i = step
        valid_acc = float(valid_correct_num) / float(valid_num)
        valid_loss /= i
        print(
            f'\nsub:{sub + 1}, fold:{fold}, epoch:{epoch + 1}\ntrain_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, '
            f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')
        subject_write.write(f'epoch:{epoch + 1}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, '
                            f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}\n')

        # TODO: 修改为比较准确率
        if valid_acc > valid_acc_max:
            valid_acc_improvements = 0
            valid_acc_max = valid_acc
            valid_acc_max_loss = valid_loss
            # 阶段一：保存了在验证集中准确率最高的模型参数
            # save_dir = cfg.params_dir + f'/sub{sub + 1}'
            params_dir_ = cfg.params_dir + '/'
            for key, val in model_params.items():
                params_dir_ += f'_{key}{val}'
            params_dir_ += f'/sub{sub + 1}'
            if not os.path.exists(params_dir_):
                os.makedirs(params_dir_)
            save_ckpt = params_dir_ + f'/fold{fold}.ckpt'
            torch.save(model.state_dict(), save_ckpt)
        else:
            valid_acc_improvements += 1
            if valid_acc_improvements >= cfg.no_improvement_times:
                break

    subject_write.close()

    return early_stopping(eeg, label, valid_ids, valid_acc_max_loss, sub, fold, model_params)


def early_stopping(eeg, label, valid_ids, valid_acc_max_loss, sub, fold, model_params=None):
    results_dir_ = cfg.results_dir + '/'
    for key, val in model_params.items():
        results_dir_ += f'_{key}{val}'
    subject_write = open(results_dir_ + f'/sub{sub + 1}/fold{fold}.txt', 'a')
    subject_write.write(f'---The early stopping stage 2.---\n')

    params_dir_ = cfg.params_dir + '/'
    for key, val in model_params.items():
        params_dir_ += f'_{key}{val}'
    save_ckpt = f'./{params_dir_}/sub{sub + 1}/fold{fold}.ckpt'
    model = Model(**model_params).to(cfg.device)
    model.load_state_dict(torch.load(save_ckpt))

    x_train, x_valid, y_train, y_valid = eeg, eeg[valid_ids], label, label[valid_ids]

    train_dataset = MyDataset(x_train, y_train)
    valid_dataset = MyDataset(x_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    # 验证集最小损失，最小损失对应准确率
    valid_loss_min = np.inf
    valid_loss_min_acc = -1.

    # ---------- train adn valid ----------
    for epoch in range(cfg.epochs_stage2):
        # 正确分类数、训练样本总数、训练集损失值
        train_correct_num, train_num, train_loss = 0, 0, 0.
        i = 0
        # ---------- train ----------
        model.train()
        for step, (data, labels) in enumerate(tqdm.tqdm(train_loader, position=0, leave=False), start=1):
            data = data.to(cfg.device)
            labels = labels.to(cfg.device)

            pred = model(data)
            loss = criterion(pred, labels)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predictions = pred.max(1)
            train_correct_num += (predictions == labels).sum()
            train_num += predictions.size(0)
            i = step
        # 训练集准确率及损失，要保存的是在验证集得到最小损失时的train_acc, train_loss
        train_acc = float(train_correct_num) / float(train_num)
        train_loss /= i

        valid_correct_num, valid_num, valid_loss = 0, 0, 0.
        # ---------- valid ----------
        model.eval()
        for step, (data, labels) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=False), start=1):
            with torch.no_grad():
                data = data.to(cfg.device)
                labels = labels.to(cfg.device)

                pred = model(data)
                loss = criterion(pred, labels)
                valid_loss += loss

                _, predictions = pred.max(1)
                valid_correct_num += (predictions == labels).sum()
                valid_num += predictions.size(0)
            i = step
        valid_acc = float(valid_correct_num) / float(valid_num)
        valid_loss /= i
        print(
            f'\nsub:{sub + 1}, fold:{fold}, epoch:{epoch + 1}\ntrain_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, '
            f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')
        subject_write.write(f'epoch:{epoch + 1}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, '
                            f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}\n')

        if valid_loss < valid_acc_max_loss:
            valid_loss_min = valid_loss
            valid_loss_min_acc = valid_acc
            torch.save(model.state_dict(), save_ckpt)
            break

    subject_write.close()

    return valid_loss_min_acc


def test_model(eeg, label, sub, fold, model_params=None):
    model = Model(**model_params).to(cfg.device)
    x_test, y_test = eeg, label

    test_dataset = MyDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # ---------- test ----------
    params_dir_ = cfg.params_dir + '/'
    for key, val in model_params.items():
        params_dir_ += f'_{key}{val}'
    save_ckpt = f'./{params_dir_}/sub{sub + 1}/fold{fold}.ckpt'
    model.load_state_dict(torch.load(save_ckpt))
    test_num, test_acc = 0, 0
    model.eval()
    for step, (data, labels) in enumerate(tqdm.tqdm(test_loader, position=0, leave=False), start=1):
        with torch.no_grad():
            data = data.to(cfg.device)
            labels = labels.to(cfg.device)
            pred = model(data)

            _, predictions = pred.max(1)
            test_acc += (predictions == labels).sum()
            test_num += predictions.size(0)
    acc = float(test_acc) / float(test_num)
    print(f'sub:{sub + 1}, fold:{fold}\ttest_acc:{acc:.4f}')
    return acc
