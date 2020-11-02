# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pandas as pd
import warnings

from WeaponLib import LoadSave

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
#
from radam import RAdamOptimizer
import tensorflow as tf
import argparse
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
np.random.seed(2019)
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
warnings.filterwarnings('ignore')
earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=2)

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=10,
                    help='choose the predicted step: 1, 10, 30, 50, 100')
parser.add_argument('--validation_fit', type=bool, default=False,
                    help='turn the valid fitting on(True) or off(False), default is False')
args = parser.parse_args()
PREDICTED_STEP = args.predictstep
valid_fit = args.validation_fit
PATH = f"..//Data//TrainedRes//sec{PREDICTED_STEP}//"

###############################################################################
def load_train_test_data():
    ls = LoadSave(PATH + "Train.pkl")
    trainData = ls.load_data()

    ls._fileName = PATH + "Test.pkl"
    testData = ls.load_data()
    return trainData, testData


swish = tf.keras.activations.swish


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


def build_model(hidden_size=18,
                #batch_size=22,
                lr=0.002,
                optimizer='adam',
                activation_1='tanh',
                activation_2='sigmoid'):
    model = Sequential()
    model.add(LSTM(hidden_size,
                   activation=activation_1,
                   recurrent_activation=activation_2,
                   return_sequences=False,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss=mean_absolute_error,
                  optimizer=Adam(lr=lr),
                  metrics=['mae'])
    if optimizer == 'radam':
        model.compile(loss=mean_squared_error,
                      optimizer=RAdamOptimizer(learning_rate=lr),
                      metrics=['mae'])
    return model

    
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_absolute_error',
                       do_probabilities=False):
    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      cv=cv,
                      n_jobs=-1,
                      scoring=scoring_fit,
                      verbose=2)
    fitted_model = gs.fit(X_train_data, y_train_data)

    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)

    return fitted_model, pred

###############################################################################
if __name__ == "__main__":
    trainData, testData = load_train_test_data()

    # Exclude
    ls = LoadSave("..//Data//TrainedRes//sec" + str(PREDICTED_STEP) + "//TestResults.pkl")
    testData["target"] = ls.load_data()

    print(f"Train shape: {trainData.shape}, Test shape: {testData.shape} before dropping nan values.")
    trainData.dropna(inplace=True)
    testData.dropna(inplace=True)

    trainData.drop("FLAG", axis=1, inplace=True)
    testData.drop("FLAG", axis=1, inplace=True)
    print(f"Train shape: {trainData.shape}, Test shape: {testData.shape} After dropping nan values.")

    numFolds = 10
    tscv = TimeSeriesSplit(n_splits=numFolds)
    folds = []
    for trainInd, validInd in tscv.split(trainData):
        folds.append([trainInd, validInd])

    # Start the time series cross validation
    score = np.zeros((numFolds, 5))
    best_hp = []
    for ind, (train, valid) in enumerate(folds):
        X_train = trainData.iloc[train].drop(["target"], axis=1).values
        X_valid = trainData.iloc[valid].drop(["target"], axis=1).values
        y_train = trainData.iloc[train]["target"].values.reshape(len(X_train), 1)
        y_valid = trainData.iloc[valid]["target"].values.reshape(len(X_valid), 1)

        # Access the normalized data
        X_sc, y_sc = MinMaxScaler(), MinMaxScaler()
        X_train = X_sc.fit_transform(X_train)
        X_valid = X_sc.transform(X_valid)  # fit 기준으로 transform하는
        X_test = X_sc.transform(testData.drop(["target"], axis=1).values)
        
        y_train = y_sc.fit_transform(y_train)
        y_valid = y_sc.transform(y_valid)
        y_test = y_sc.transform(testData["target"].values.reshape(len(X_test), 1))
        
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Start training the model
        param_grid = {
            'hidden_size': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            'batch_size': [10, 14, 18, 22, 24, 28, 32, 64, 128, 256, 512],
            'lr': [1e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            'optimizer': ['adam'], #['adam', 'radam'],
            'activation_1': ['tanh'], #['tanh', swish, mish],
            'activation_2': ['sigmoid', mish] #['sigmoid', swish, mish]
        }
        #bestparam = {
        #               'activation_1': 'tanh',
        #               'activation_2': mish,
        #               'batch_size': 22,
        #               'hidden_size': 18,
        #               'lr': 0.001,
        #               'optimizer': 'adam'
        #            }

        # search the best hp
        model = KerasRegressor(build_fn=build_model, verbose=0)
        if valid_fit:
            model.fit(X_train, y_train,
                      epochs=500, batch_size=22, ###TODO
                      validation_data=(X_valid, y_valid), verbose=1,
                      shuffle=False, callbacks=[earlyStopping])
        model, y_pred = algorithm_pipeline(X_train, X_test,
                                           y_train, y_test,
                                           model, param_grid)

        #with open('result.txt', 'a') as fp:
        #    fp.write(f'({model.best_score_}, {model.best_params_})\n')
        print('\n=====')
        mdl_bs = model.best_score_
        mdl_bs = y_sc.inverse_transform(mdl_bs.reshape((-1, 1)))
        print(mdl_bs)
        print(model.best_params_)
        print('=====\n')

        y_test = y_sc.inverse_transform(y_test)
        y_pred = y_pred.reshape((-1, 1))
        y_pred = y_sc.inverse_transform(y_pred)
        y_pred[y_pred < 1] = 0

        score[ind, 0] = ind + 1
        score[ind, 1] = mdl_bs
        score[ind, 2] = r2_score(y_test, y_pred)
        score[ind, 3] = sklearn.metrics.mean_absolute_error(y_test, y_pred)
        score[ind, 4] = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
        best_hp.append(str(model.best_params_))
        
        start, end = 0, len(y_test)
        plt.figure(figsize=(16, 10))
        plt.plot(y_pred[start:end], linewidth=2, linestyle="-", color="r")
        plt.plot(y_test[start:end], linewidth=2, linestyle="-", color="b")
        plt.legend(["Prediction", "Ground Truth"])
        plt.xlim(0, end - start)
        plt.ylim(-500, 2600)
        plt.grid(True)
        if not os.path.exists('..//Plots2'):
            os.makedirs('..//Plots2')
        plt.savefig(f"..//Plots2//PredictedStepTest_{PREDICTED_STEP}_folds_{ind + 1}_.png",
                    dpi=50, bbox_inches="tight")
        plt.close("all")

    score = pd.DataFrame(score, columns=['fold', 'best_score', 'R-square', 'MAE', 'RMSE'])
    best_hp = pd.DataFrame(best_hp, columns=['best_params'])

    result_table = pd.concat([score, best_hp], axis=1)
    print(result_table)

    # save
    #score.to_pickle("score.pkl")
    result_table.to_pickle('..//Plots2//result_table.pkl')
    result_table.to_csv('..//Plots2//result_table.csv')

    # #load
    # df = pd.read_pickle("score.pkl")
    # df = pd.read_csv('sample.csv')
