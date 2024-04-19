import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from Tools. model_training import get_best_params
from Tools.data_processing import Training_Loader, Testing_Loader, Data_Pre
from networks.net import gtnet
import time
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.base import clone
warnings.filterwarnings('ignore')
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
torch.set_num_threads(3)

filename = "./logfiles/MiPM.log"
logging.basicConfig(filename=filename, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S', filemode='w', level=logging.INFO)
prediction_horizon = 1
scaler_cols, scaler_target, X_train, y_train, X_test, y_test = Data_Pre(
    horizon=prediction_horizon)
train_len = int(X_train.shape[0] * 0.75)
val_len = X_train.shape[0] - train_len
train_loader, val_loader = Training_Loader(
    X_train[:train_len], y_train[:train_len], X_train[train_len:], y_train[train_len:])
test_loader = Testing_Loader(X_test, y_test)
param_grid = {
    'dropout': [0.1 * i for i in range(1, 5)],
    'leaky_rate': [0.1 * i for i in range(1, 5)],
    'propalpha': [0.1 * i for i in range(1, 5)],
}
add_bf = 0
base_estimator = gtnet(device_name=device_name).to(device)
best_params = get_best_params(
    base_estimator, param_grid, X_train, y_train, device)
logging.info('best params: propalpha: {}, leaky_rate:{}, dropout: {}'.format(
    best_params['propalpha'], best_params['leaky_rate'], best_params['dropout']))
model = clone(clone(base_estimator).set_params(**best_params))
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
epochs = 100
loss = nn.MSELoss().to(device)
min_val_loss = 9999
n_samples = 0
depth = 64
logging.info(model)
logging.info("Window = "+str(depth))
logging.info("Horizon = "+str(prediction_horizon))
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
        output1, output2 = model(batch_x.to(device), batch_y_h)
        if add_bf:
            output1 = torch.squeeze(output1)
            output2 = torch.squeeze(output2)
            l = (loss(output1, batch_y) +
                 loss(output2, batch_y)) / 2.0
        else:
            output1 = torch.squeeze(output1)
            l = loss(output1, batch_y)
        l.backward()
        mse_train += l.item() * batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    model.eval()
    with torch.no_grad():
        mse_val = 0
        total_loss = 0
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
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]
    if min_val_loss > mse_val ** 0.5:
        min_val_loss = mse_val ** 0.5
        filename = "./MiPM.pt"
        torch.save(model.state_dict(), filename)
    epoch_end = time.monotonic()
    if epoch % 10 == 0:
        logging.info(
            '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (epoch_end - epoch_start), (mse_train/train_len)**0.05, (mse_val/val_len)**0.05))
train_end = time.monotonic()

# traing time
logging.info("MiPM training time: {:.4f}".format(train_end - train_start))
# train metrics caculate
filename = "./MiPM.pt"
model.load_state_dict(torch.load(filename))
model.eval()
with torch.no_grad():
    preds = []
    true = []
    for batch_x, batch_y_h, batch_y in train_loader:
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
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
preds = np.concatenate(preds)
true = np.concatenate(true)
true = scaler_target.inverse_transform(true)
preds = scaler_target.inverse_transform(preds)
mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
r2 = r2_score(true, preds)
mape = np.mean(np.abs((preds - true) / true)) * 100
t_mean = np.mean(true)
logging.info('Train data result:')
logging.info('Train RMSE: {:.4f}, Train R2: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}'.format(
    mse**0.5, r2, mae, mape))
result = np.concatenate((true, preds), axis=1)
# test metrics caculate
filename = "./MiPM.pt"
model.load_state_dict(torch.load(filename))
model.eval()
with torch.no_grad():
    preds = []
    true = []
    for batch_x, batch_y_h, batch_y in test_loader:
        batch_y = batch_y.to(device)
        batch_y_h = batch_y_h.to(device)
        batch_y_h = torch.unsqueeze(batch_y_h, dim=1)
        batch_y_h = batch_y_h.transpose(2, 3)
        batch_x = torch.unsqueeze(batch_x, dim=1)
        batch_x = batch_x.transpose(2, 3)
        output, _ = model(batch_x.to(device), batch_y_h.to(device))
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
preds = np.concatenate(preds)
true = np.concatenate(true)
true = scaler_target.inverse_transform(true)
preds = scaler_target.inverse_transform(preds)
mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
r2 = r2_score(true, preds)
mape = np.mean(np.abs((preds - true) / true)) * 100
t_mean = np.mean(true)
print(mean_absolute_percentage_error(true, preds, multioutput='raw_values'))
logging.info('Test data result:')
logging.info('Test RMSE: {:.4f}, Test R2: {:.4f}, Test MAE: {:.4f}, Test MAPE: {:.4f}'.format(
    mse**0.5, r2, mae, mape))
