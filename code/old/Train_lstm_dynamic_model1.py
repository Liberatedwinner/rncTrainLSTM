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
# import tensorflow_addons as tfa # pip install tensorflow-addons
# import math

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
np.random.seed(2019)
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
warnings.filterwarnings('ignore')
earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=2)

parser = argparse.ArgumentParser()
# parser.add_argument('--activation1', type=str, default='tanh',
#                     help='choose the activation function instead of tanh: swish, mish')
# parser.add_argument('--activation2', type=str, default='sigmoid',
#                     help='choose the activation function instead of sigmoid: swish, mish')
parser.add_argument('--predictstep', type=int, default=1,
                    help='choose the predicted step: 1, 10, 30, 50, 100')
args = parser.parse_args()
PREDICTED_STEP = args.predictstep
PATH = f"..//Data//TrainedRes//sec{PREDICTED_STEP}//"
# activation1 = args.activation1
# activation2 = args.activation2

# PREDICTED_STEP = args.predictstep
# if PREDICTED_STEP == 10:
#     PATH = f"..//Data//TrainedRes//sec{PREDICTED_STEP}//"
# elif PREDICTED_STEP == 30:
#     PATH = "..//Data//TrainedRes//sec30//"
# elif PREDICTED_STEP == 50:
#     PATH = "..//Data//TrainedRes//sec50//"
# elif PREDICTED_STEP == 100:
#     PATH = "..//Data//TrainedRes//sec100//"
# else:
#     PATH = "..//Data//TrainedRes//sec1//"

###############################################################################
def load_train_test_data():
    ls = LoadSave(PATH + "Train.pkl")
    trainData = ls.load_data()

    ls._fileName = PATH + "Test.pkl"
    testData = ls.load_data()
    return trainData, testData


# def plot_history(history, result_dir):
#     plt.figure()
#     plt.plot(history.history['loss'], marker='.')
#     plt.plot(history.history['val_loss'], marker='.')
#     plt.title('Model Mean Absoluted Error')
#     plt.xlabel('epoch')
#     plt.ylabel('MAE')
#     plt.grid()
#     plt.legend(['mae', 'val_loss'], loc='upper right')
#     plt.savefig(result_dir, dpi=500, bbox_inches="tight")
#     plt.close()

# def swish(x):
#     swish_value = x * sigmoid(x)
#     if swish_value < 1:
#         swish_value = 1
#     return swish_value
#
# def mish(x):
#     mish_value = x * math.tanh(math.log((1 + math.exp(x))))
#     if mish_value < 1:
#         mish_value = 1
#     return mish_value


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


swish = tf.keras.activations.swish

# if activation1 == 'swish':
#     activation_f1 = swish
# elif activation1 == 'mish':
#     activation_f1 = mish
#
#
# if activation2 == 'swish':
#     activation_f2 = swish
# elif activation2 == 'mish':
#     activation_f2 = mish


def build_model(hidden_size=18,
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
                  metrics=['mean_square_error'])
    if optimizer == 'radam':
        model.compile(loss=mean_absolute_error,
                      optimizer=RAdamOptimizer(learning_rate=lr),
                      metrics=['mean_square_error'])
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
    score = np.zeros((numFolds, 6))
    # score = np.zeros((numFolds, 4))
    for ind, (train, valid) in enumerate(folds):
        X_train = trainData.iloc[train].drop(["target"], axis=1).values
        X_valid = trainData.iloc[valid].drop(["target"], axis=1).values
        y_train = trainData.iloc[train]["target"].values.reshape(len(X_train), 1)
        y_valid = trainData.iloc[valid]["target"].values.reshape(len(X_valid), 1)

        # Access the normalized data
        X_sc, y_sc = MinMaxScaler(), MinMaxScaler()
        X_train = X_sc.fit_transform(X_train)
        X_valid = X_sc.transform(X_valid) #fit 기준으로 transform하는
        X_test = X_sc.transform(testData.drop(["target"], axis=1).values)

        y_train = y_sc.fit_transform(y_train)
        y_valid = y_sc.transform(y_valid)
        y_test = y_sc.transform(testData["target"].values.reshape(len(X_test), 1))

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Start training the model
        model = build_model()
        history = model.fit(X_train, y_train, epochs=500, batch_size=64,
                            validation_data=(X_valid, y_valid), verbose=1,
                            shuffle=False, callbacks=[earlyStopping])
        model.evaluate(X_test, y_test, verbose=0)

        y_valid_pred = model.predict(X_valid)
        y_valid, y_valid_pred = y_sc.inverse_transform(y_valid), y_sc.inverse_transform(y_valid_pred)
        y_valid_pred[y_valid_pred < 1] = 0

        y_test_pred = model.predict(X_test)
        y_test, y_test_pred = y_sc.inverse_transform(y_test), y_sc.inverse_transform(y_test_pred)
        y_test_pred[y_test_pred < 1] = 0

        score[ind, 0] = ind
        score[ind, 1] = sklearn.metrics.mean_absolute_error(y_valid, y_valid_pred)
        score[ind, 2] = np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_valid_pred))
        score[ind, 3] = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
        score[ind, 4] = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))
        score[ind, 5] = r2_score(y_test, y_test_pred)


        with open('result.txt', 'a') as fp:
            fp.write(f'({model.best_score_}, {model.best_params_})\n')

        print(model.best_score_)
        print(model.best_params_)


        start, end = 0, len(y_test)
        plt.figure(figsize=(16, 10))
        plt.plot(y_test_pred[start:end], linewidth=2, linestyle="-", color="r")
        plt.plot(y_test[start:end], linewidth=2, linestyle="-", color="b")
        plt.legend(["Prediction", "Ground Truth"])
        plt.xlim(0, end - start)
        plt.ylim(-500, 2600)
        plt.grid(True)
        if not os.path.exists('..//Plots'):
            os.makedirs('..//Plots')
        # plt.savefig("..//Plots//PredictedStepTest_" + str(PREDICTED_STEP) + "_folds_" + str(ind + 1) + "_Original.png",
        #             dpi=50, bbox_inches="tight")
        plt.savefig(f"..//Plots//PredictedStepTest_{PREDICTED_STEP}_folds_{ind + 1}_Original.png",
                    dpi=50, bbox_inches="tight")
        plt.close("all")
    score = pd.DataFrame(score, columns=["fold", "validMAE", "validRMSE", "testMAE", "testRMSE", 'testRsqr'])
    print(score)

    #save
    score.to_pickle("score.pkl")

    # #load
    # df = pd.read_pickle("score.pkl")