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
import keras.wrappers


rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
np.random.seed(2019)
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
warnings.filterwarnings('ignore')
earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=2)

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=1,
                    help='choose the predicted step: 1, 10, 30, 50, 100')
args = parser.parse_args()
PREDICTED_STEP = args.predictstep
PATH = f"..//Data//TrainedRes//sec{PREDICTED_STEP}//"

###############################################################################
def load_train_test_data():
    ls = LoadSave(PATH + "Train.pkl")
    trainData = ls.load_data()

    ls._fileName = PATH + "Test.pkl"
    testData = ls.load_data()
    return trainData, testData


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


swish = tf.keras.activations.swish


def build_model(hidden_size=20,
                batch_size=128,
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
                  metrics=['cosine_similarity'])
    if optimizer == 'radam':
        model.compile(loss=mean_absolute_error,
                      optimizer=RAdamOptimizer(learning_rate=lr),
                      metrics=['cosine_similarity'])
    # model.fit(X_train, y_train, epochs=500, batch_size=batch_size,
    #                     validation_data=(X_valid, y_valid), verbose=1,
    #                     shuffle=False, callbacks=[earlyStopping])
    return model


def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
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
    # score = np.zeros((numFolds, 6))
    score = np.zeros((numFolds, 4))
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
            # 'hidden_size': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            'hidden_size': [10, 14, 18, 22, 26, 30],
            'batch_size': [32, 64, 128, 256, 512],
            'lr': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            'optimizer': ['adam', 'radam'],
            'activation_1': ['tanh', swish, mish],
            'activation_2': ['sigmoid', swish, mish]
        }

        # search the best hp
        model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model, verbose=0)
        model, y_pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, param_grid)

        with open('result.txt', 'a') as fp:
            fp.write(f'({model.best_score_}, {model.best_params_})\n')

        print(model.best_score_)
        print(model.best_params_)
        y_test = y_sc.inverse_transform(y_test)
        y_pred = y_sc.inverse_transform(y_pred)
        y_pred[y_pred < 1] = 0

        score[ind, 0] = ind
        score[ind, 1] = model.best_score_
        score[ind, 2] = r2_score(y_test, y_pred)
        score[ind, 3] = model.best_params_

        start, end = 0, len(y_test)
        plt.figure(figsize=(16, 10))
        plt.plot(y_pred[start:end], linewidth=2, linestyle="-", color="r")
        plt.plot(y_test[start:end], linewidth=2, linestyle="-", color="b")
        plt.legend(["Prediction", "Ground Truth"])
        plt.xlim(0, end - start)
        plt.ylim(-500, 2600)
        plt.grid(True)
        if not os.path.exists('..//Plots'):
            os.makedirs('..//Plots')
        plt.savefig(f"..//Plots//PredictedStepTest_{PREDICTED_STEP}_folds_{ind + 1}_Original.png",
                    dpi=50, bbox_inches="tight")
        plt.close("all")
    score = pd.DataFrame(score, columns=['fold', 'best_score', 'R-square', 'best_params'])
    print(score)

    # save
    score.to_pickle("score.pkl")

    # #load
    # df = pd.read_pickle("score.pkl")
