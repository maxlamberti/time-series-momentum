#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


# In[3]:


from utils.callbacks import GetBest
from utils.models import MultiLayerPerceptron
from utils.data_loaders import load_clc_db_records
from utils.features import construct_features_batch
from utils.data_handling import merge_asset_data, split_by_date
from utils.loss_functions import return_loss, sharpe_loss, return_loss_dummy, sharpe_loss_dummy


# In[4]:


from data.data_config import ASSETS_TO_USE


# In[5]:


config = tf.ConfigProto(device_count = {'GPU': 0 , 'CPU': 3}) 
sess = tf.Session(config=config) 
K.set_session(sess)


# In[6]:


RAD_DATA_PATH = '../data/clc/rad/'
FED_DATA_PATH = '../data/DFF.csv'


# ## Load Asset Data & Prepare Features

# In[7]:


# load asset data from clc database
clc = load_clc_db_records(RAD_DATA_PATH, FED_DATA_PATH, ASSETS_TO_USE)
clc = construct_features_batch(clc)
df = merge_asset_data(clc, create_time_asset_index=True)
df.dropna(inplace=True)


# In[8]:


# create windowed subset of the data
date_breakpoints = [datetime(1990, 1, 1)] + [datetime(year, 1, 1) for year in range(1995, 2021)]
date_breakpoints = [datetime(1990, 1, 1)] + [datetime(year, 1, 1) for year in range(1995, 2021, 5)]
print(date_breakpoints)
data_set = split_by_date(df, date_breakpoints)


# In[9]:


data_set[0].head()


# In[ ]:


data_set[0].tail()


# ## Train Multilayer Perceptron DMN

# In[1]:


# strategy parameters
target_vola = 0.15
use_expanding_window = True # if False uses rolling window during training

# data parameters
feature_labels = [
   'Norm_Returns_Daily', 'Norm_Returns_Monthly', 'Norm_Returns_Quarterly',
   'Norm_Returns_Semiannually', 'Norm_Returns_Annually', 'MACD_8_24',
   'MACD_16_48', 'MACD_32_96','Sigma_Norm'
]  #'Binary_MACD_8_24', 'Binary_MACD_16_48', 'Binary_MACD_32_96'
target_labels = ['Next_Returns_Daily', 'Sigma']  # need multi target for custom loss function

# available optimizers
adagrad = optimizers.Adagrad()
sgd = optimizers.SGD(momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.008, beta_1=0.9, beta_2=0.999, amsgrad=False)

# available callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')
return_best_model = GetBest(monitor='val_loss', verbose=0, mode='max')
mcp_save = ModelCheckpoint('../models/best_mlp.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# MLP network parameters
epochs = 100
batch_size = 2 ** 6
input_dropout = 0
hidden_neurons = [6, 5, 3]
hidden_layers = len(hidden_neurons)
hidden_dropout = 0.35
hidden_activation = 'relu'
loss = sharpe_loss
optimizer = adam
callbacks = [
    early_stopping,
    return_best_model
]
shuffle = False  # shuffle training data on each epoch
normalize_features = True


# In[ ]:


# # define an objective function

# # construct features and targets
# data_series = data_set[0]
# val_series = data_set[1]
# scaler = StandardScaler()
# scaler.fit(data_series[feature_labels].values)
# X = scaler.transform(data_series[feature_labels].values)
# X = data_series[feature_labels].values
# y = data_series[target_labels].values
# X_val = scaler.transform(val_series[feature_labels].values)
# y_val = val_series[target_labels].values


# def hyperopt_objective(args, X=X, y=y, X_val=X_val, y_val=y_val):
    
#     print(args)

#     # define model
#     model = construct_mlp_model(**args)
#     opti = optimizers.Adam(lr=args['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False)
#     model.compile(loss=sharpe_loss, optimizer=opti)
#     earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
#     history = model.fit(
#         X, y,
#         batch_size=batch_size,
#         epochs=epochs,
#         verbose=0,
#         validation_data=(X_val, y_val),
#         shuffle=False,
#         callbacks=[earlyStopping]
#     )
    
#     return np.mean(history.history['val_loss'][2:-5])
    
# # define a search space
# from hyperopt import hp
# activation_funcs = ['sigmoid', 'relu']
# space = {
#     'input_shape': len(feature_labels),
#     'lr': hp.loguniform('lr', -10, 0),
#     'input_activation': hp.choice('input_activation', activation_funcs),
#     'hidden_activation': hp.choice('hidden_activation', activation_funcs),
#     'hidden_layers': hp.quniform('hidden_layers', 1, 3, 1),
#     'hidden_neurons': hp.quniform('hidden_neurons', 1, 6, 1),
#     'hidden_dropout': hp.uniform('hidden_dropout', 0, 0.5),
#     'input_dropout': hp.uniform('input_dropout', 0, 0.5)
# }

# # minimize the objective over the space
# from hyperopt import fmin, tpe, space_eval
# best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=100)


# In[ ]:


# declare MLP model
MLP = MultiLayerPerceptron(
    num_features=len(feature_labels), 
    input_dropout=input_dropout, 
    hidden_layers=hidden_layers, 
    hidden_neurons=hidden_neurons, 
    hidden_dropout=hidden_dropout, 
    hidden_activation=hidden_activation,
    loss=loss,
    optimizer=optimizer,
    normalize_features=normalize_features
)
MLP.summary()


# In[ ]:


turnover, backtest_returns, backtest_excess_returns = [], [], []
temp_series = pd.DataFrame()  # use this to expand rolling window

for data_idx, data_series in enumerate(data_set[:-1]):
    
    # training time window
    epoch_start = data_series.Date_Col.min()
    epoch_end = data_series.Date_Col.max()
    
    # if exanding window concat previous data
    if use_expanding_window:
        data_series = pd.concat([data_series, temp_series])
        temp_series = data_series
    
    # use last year of windowed data as validation set
    val_series = data_series[data_series['Date_Col'] >= (epoch_end - timedelta(days=364))]
    train_series = data_series[data_series['Date_Col'] < (epoch_end - timedelta(days=364))]
    backtest_data_set = data_set[data_idx + 1]  # backtest on next window 
    
    # construct feature and target matrices
    X = train_series[feature_labels].values
    y = train_series[target_labels].values
    X_val = val_series[feature_labels].values
    y_val = val_series[target_labels].values

    MLP.fit(
        X, y,
        X_val, y_val,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        shuffle=shuffle,
        callbacks=callbacks
    )
    
    break
    