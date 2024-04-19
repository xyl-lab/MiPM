from sklearn.model_selection import KFold, ParameterGrid
from Tools.data_processing import Training_Loader
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
import time
import torch
import torch.nn as nn
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import gc


def refit(model, train_len, val_len, train_loader, val_loader, device):
    epochs = 100
    min_val_loss = 9999
    loss_function = nn.MSELoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
    train_start = time.monotonic()
    for i in range(epochs):
        mse_train = 0
        model.train()
        for batch_x, batch_y_h, batch_y in train_loader:
            opt.zero_grad()
            y_pred = model(batch_x.to(device), batch_y_h.to(device))
            loss = loss_function(y_pred, batch_y.to(device))
            loss.backward()
            mse_train += loss.item() * batch_x.shape[0]
            opt.step()
        epoch_scheduler.step()
        model.eval()
        with torch.no_grad():
            mse_val = 0
            for batch_x, batch_y_h, batch_y in val_loader:
                output = model(batch_x.to(device), batch_y_h.to(device))
                mse_val += loss_function(output, batch_y.to(device)
                                         ).item() * batch_x.shape[0]
        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            filename = ".././MTSGAM.pt"
            torch.save(model.state_dict(), filename)
        if i % 10 == 0:
            logging.info("Iter: " + str(i) + " train: " + str((mse_train / train_len) ** 0.5) + " val: " + str(
                (mse_val / val_len) ** 0.5))
    train_end = time.monotonic()
    logging.info("MTSGAM training time: {:.4f}".format(
        train_end - train_start))


def fit(model, train_len, val_len, train_loader, val_loader, device):
    epochs = 100
    min_val_loss = 9999
    loss_function = nn.MSELoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
    train_start = time.monotonic()
    for epoch in range(epochs):
        mse_train = 0
        epoch_start = time.monotonic()
        model.train()
        for batch_x, batch_y_h, batch_y in train_loader:
            opt.zero_grad()
            batch_y = batch_y.to(device)
            batch_y_h = batch_y_h.to(device)
            batch_y_h = torch.unsqueeze(batch_y_h, dim=1)
            batch_x = torch.unsqueeze(batch_x, dim=1)
            batch_x = batch_x.transpose(2, 3)
            batch_y_h = batch_y_h.transpose(2, 3)
            y_pred1, y_pred2 = model(batch_x.to(device), batch_y_h)
            y_pred1 = torch.squeeze(y_pred1)
            loss = loss_function(y_pred1, batch_y)
            mse_train += loss.item() * batch_x.shape[0]
            loss.backward()
            opt.step()
        epoch_scheduler.step()
        model.eval()
        with torch.no_grad():
            mse_val = 0
            for batch_x, batch_y_h, batch_y in val_loader:
                batch_y = batch_y.to(device)
                batch_y_h = batch_y_h.to(device)
                batch_x = torch.unsqueeze(batch_x, dim=1)
                batch_x = batch_x.transpose(2, 3)
                batch_y_h = torch.unsqueeze(batch_y_h, dim=1)
                batch_y_h = batch_y_h.transpose(2, 3)
                output, _ = model(batch_x.to(device), batch_y_h.to(device))
                output = torch.squeeze(output)
                if len(output.shape) == 1:
                    output = output.unsqueeze(dim=0)
                mse_val += loss_function(output,
                                         batch_y).item() * batch_x.shape[0]
        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
        epoch_end = time.monotonic()
        if epoch % 10 == 0:
            logging.info(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | val_loss {:5.4f}'.format(
                    epoch, (epoch_end - epoch_start), (mse_train / train_len) ** 0.5, (mse_val / val_len) ** 0.5))
    train_end = time.monotonic()
    # traing time
    logging.info("MTSGAM training time: {:.1f}".format(
        train_end - train_start))
    return min_val_loss


def cross_verification(base_estimator, parameters, X_train, y_train, device):
    skfolds = KFold(n_splits=4)
    local_loss = []
    for train_index, test_index in skfolds.split(X_train, y_train):
        estimator = clone(base_estimator)
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)
        estimator = clone(estimator.set_params(**cloned_parameters))
        estimator.to(device)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train[test_index])
        train_len = X_train_folds.shape[0]
        val_len = X_test_fold.shape[0]
        train_loader, val_loader = Training_Loader(
            X_train_folds, y_train_folds, X_test_fold, y_test_fold)

        local_min_val_loss = fit(
            estimator, train_len, val_len, train_loader, val_loader, device)
        local_loss.append(local_min_val_loss)
        del estimator
        gc.collect()
    return np.mean(local_loss)


def get_best_params(estimator, param_grid, X_train, y_train, device):
    global_val_loss = []
    candidate_params = list(ParameterGrid(param_grid))
    base_estimator = clone(estimator)
    for cand_idx, parameters in enumerate(candidate_params):
        logging.info('cuurent parameters index: {}'.format(cand_idx))
        local_val_loss = cross_verification(
            base_estimator, parameters, X_train, y_train, device)
        global_val_loss.append(local_val_loss)
        torch.cuda.empty_cache()
    for cand_idx, parameters in enumerate(candidate_params):
        logging.info('val loss: {}'.format(global_val_loss[cand_idx]))
    index = np.argmin(global_val_loss)
    best_parameters = candidate_params[index]
    return best_parameters
