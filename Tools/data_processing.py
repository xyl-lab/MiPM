import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import gc
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.model_selection import GridSearchCV


def Data_Pre(window=64, batch_size=128, horizon=1, dpath='./Dataset/stress.csv', tpath='./Dataset/targets.csv'):

    x_data = pd.read_csv(dpath,
                         index_col='Datetime', parse_dates=['Datetime'])
    y_data = pd.read_csv(tpath,
                         index_col='Datetime', parse_dates=['Datetime'])
    # 扩大倍数
    # y_data.iloc[:, :3] *= 1000
    # 使用连续下沉量
    # for i in range(1, y_data.shape[0]):
    #     y_data.values[i, 0] += y_data.values[i-1, 0]

    cols = x_data.columns
    target = y_data.columns
    raw_data = pd.merge(x_data, y_data, left_index=True, right_index=True)
    tlen = len(target)
    dlen = len(cols)
    L = len(x_data)
    train_size = int(0.6*L)
    val_size = int(0.2*L)
    test_size = L - train_size - val_size

    data_train = raw_data.iloc[:train_size + val_size]
    data_test = raw_data.iloc[train_size+val_size:]

    scaler_cols = MinMaxScaler(feature_range=(
        0, 1)).fit(data_train[cols].values)
    data_train_cols_scale = scaler_cols.transform(data_train[cols].values)
    data_test_cols_scale = scaler_cols.transform(data_test[cols].values)
    data_train_cols_scale = pd.DataFrame(data_train_cols_scale)
    data_test_cols_scale = pd.DataFrame(data_test_cols_scale)

    scaler_target = MinMaxScaler(feature_range=(
        0, 1)).fit(data_train[target].values)
    data_train_target_scale = scaler_target.transform(
        data_train[target].values)
    data_test_target_scale = scaler_target.transform(data_test[target].values)
    data_train_target_scale = pd.DataFrame(data_train_target_scale)
    data_test_target_scale = pd.DataFrame(data_test_target_scale)

    # train
    X1 = np.zeros((train_size + val_size, window, len(cols)))
    y_his1 = np.zeros((train_size + val_size, window, tlen))
    y1 = np.zeros((train_size + val_size, tlen))

    for i, name in enumerate(data_train_cols_scale.columns):
        for j in range(window):
            X1[:, j, i] = data_train_cols_scale[name].shift(
                window - j - 1).fillna(method="bfill")
    for j in range(window):
        y_his1[:, j, :] = data_train_target_scale.shift(
            window - j - 1).fillna(method="bfill")
    y1 = data_train_target_scale.shift(- window -
                                       horizon+1).fillna(method="bfill")

    X_train = X1[window-1:-horizon]
    y_his_train = y_his1[window-1:-horizon]
    y_train = y1[:-window-horizon+1]

    del X1, y1, y_his1, data_train_cols_scale, data_train_target_scale
    gc.collect()

    # test
    X3 = np.zeros((test_size, window, len(cols)))
    y_his3 = np.zeros((test_size, window, tlen))
    y3 = np.zeros((test_size, tlen))

    for i, name in enumerate(data_test_cols_scale.columns):
        for j in range(window):
            X3[:, j, i] = data_test_cols_scale[name].shift(
                window - j - 1).fillna(method="bfill")
    for j in range(window):
        y_his3[:, j, :] = data_test_target_scale.shift(
            window - j - 1).fillna(method="bfill")
    y3 = data_test_target_scale.shift(- window -
                                      horizon+1).fillna(method="bfill")

    X_test = X3[window-1:-horizon]
    y_his_test = y_his3[window-1:-horizon]
    y_test = y3[:-window-horizon+1]

    del X3, y3, y_his3, data_test_cols_scale, data_test_target_scale
    gc.collect()
    # X_train 自变量，y_his历史预测值，y_tain预测值对应的真实值
    X_train_t = torch.Tensor(X_train)
    y_his_train_t = torch.Tensor(y_his_train)
    y_train_t = torch.Tensor(y_train.values)
    X_test_t = torch.Tensor(X_test)
    y_his_test_t = torch.Tensor(y_his_test)
    y_test_t = torch.Tensor(y_test.values)
    X_train = torch.cat((X_train_t, y_his_train_t), dim=2)
    y_train = y_train_t
    X_test = torch.cat((X_test_t, y_his_test_t), dim=2)
    y_test = y_test_t
    return scaler_cols, scaler_target, X_train, y_train, X_test, y_test, dlen, tlen


def Training_Loader(X_train_folds, y_train_folds, X_test_fold, y_test_fold, batch_size=32, dsize=38):
    X_train = X_train_folds[:, :, :dsize]
    y_his_train = X_train_folds[:, :, dsize:]

    X_test = X_test_fold[:, :, :dsize]
    y_his_test = X_test_fold[:, :, dsize:]
    train_loader = DataLoader(TensorDataset(
        X_train, y_his_train, y_train_folds), shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(
        X_test, y_his_test, y_test_fold), shuffle=True, batch_size=batch_size)
    return train_loader, val_loader


def Testing_Loader(X, y, batch_size=32, dsize=38):
    X_test = X[:, :, :dsize]
    y_his_test = X[:, :, dsize:]
    test_loader = DataLoader(TensorDataset(
        X_test, y_his_test, y), batch_size=batch_size)
    return test_loader
