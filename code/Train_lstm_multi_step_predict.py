#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pandas as pd
import warnings

from WeaponLib import ReduceMemoryUsage
from WeaponLib import LoadSave

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from radam import RAdamOptimizer
from radam2 import RAdam
import tensorflow as tf
import tensorflow_addons as tfa # pip install tensorflow-addons
import argparse

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
np.random.seed(2019)
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
warnings.filterwarnings('ignore')
earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=2)
lr = 0.005
metric = 'cosine_similarity'

PREDICTED_STEP = 10
if PREDICTED_STEP == 10:
    PATH = "..//Data//TrainedRes//sec10//"
elif PREDICTED_STEP == 50:
    PATH = "..//Data//TrainedRes//sec50//"
elif PREDICTED_STEP == 100:
    PATH = "..//Data//TrainedRes//sec100//"
elif PREDICTED_STEP == 30:
    PATH = "..//Data//TrainedRes//sec30//"
else:
    PATH = "..//Data//TrainedRes//sec1//"


swish = tf.keras.activations.swish
mish = tfa.activations.mish

parser = argparse.ArgumentParser()
parser.add_argument('--activation1', type=str, default='tanh',
                    help='choose the activation function instead of tanh: swish, mish')
parser.add_argument('--activation2', type=str, default='sigmoid',
                    help='choose the activation function instead of sigmoid: swish, mish')
args = parser.parse_args()
activation1 = args.activation1
activation2 = args.activation2

if activation1 == 'swish':
    activation1 = swish
elif activation1 == 'mish':
    activation1 = mish

if activation2 == 'swish':
    activation2 = swish
elif activation2 == 'mish':
    activation2 = mish
###############################################################################
def load_train_test_data():
    ls = LoadSave(PATH + "Train.pkl")
    trainData = ls.load_data()
    
    ls._fileName = PATH + "Test.pkl"
    testData = ls.load_data()
    return trainData, testData


def plot_history(history, result_dir):
    plt.figure()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('Model Mean Absoluted Error')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.grid()
    plt.legend(['mae', 'val_loss'], loc='upper right')
    plt.savefig(result_dir, dpi=500, bbox_inches="tight")
    plt.close()


def save_history(hist, metric):
    loss = hist.history['loss']
    acc = hist.history[metric]
    val_loss = hist.history['val_loss']
    val_acc = hist.history[f'val_{metric}']
    nb_epoch = len(acc)

    with open(f'result{lr}.txt', 'a') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


###############################################################################
if __name__ == "__main__":
    trainData, testData = load_train_test_data()


    ls = LoadSave("..//Data//TrainedRes//sec" + str(PREDICTED_STEP) + "//TestResults.pkl")
    testData["target"] = ls.load_data()
    
    print("Train shape:{}, Test shape:{} before dropping nan values.".format(trainData.shape, testData.shape))
    trainData.dropna(inplace=True)
    testData.dropna(inplace=True)
    
    trainData.drop("FLAG", axis=1, inplace=True)
    testData.drop("FLAG", axis=1, inplace=True)
    print("Train shape:{}, Test shape:{} After dropping nan values.".format(trainData.shape, testData.shape))
    
    numFolds = 10
    tscv = TimeSeriesSplit(n_splits=numFolds)
    folds = []
    for trainInd, validInd in tscv.split(trainData):
        folds.append([trainInd, validInd])
    
    # Start the time series cross validation
    score = np.zeros((numFolds, 5))
    for ind, (train, valid) in enumerate(folds):
        X_train, X_valid = trainData.iloc[train].drop(["target"], axis=1).values, trainData.iloc[valid].drop(["target"], axis=1).values
        y_train, y_valid = trainData.iloc[train]["target"].values.reshape(len(X_train), 1), trainData.iloc[valid]["target"].values.reshape(len(X_valid), 1)
        
        # Access the normalized data
        X_sc, y_sc = MinMaxScaler(), MinMaxScaler()
        X_train = X_sc.fit_transform(X_train)
        X_valid = X_sc.transform(X_valid)
        X_test = X_sc.transform(testData.drop(["target"], axis=1).values)
        
        y_train = y_sc.fit_transform(y_train)
        y_valid = y_sc.transform(y_valid)
        y_test = y_sc.transform(testData["target"].values.reshape(len(X_test), 1))

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        filename = f'checkpoint-fold {folds} - lr={lr}'
        checkpoint = ModelCheckpoint(filename,  # file명을 지정합니다
                                     monitor='val_loss',  # val_loss 값이 개선되었을때 호출됩니다
                                     verbose=1,  # 로그를 출력합니다
                                     save_best_only=True,  # 가장 best 값만 저장합니다
                                     mode='auto'  # auto는 알아서 best를 찾습니다. min/max
                                     )

        # Start training the model
        #model = Sequential([Activation("relu")])
        model = Sequential()
        model.add(LSTM(20,
                       activation=activation1,
                       recurrent_activation=activation2,
                       return_sequences=False,
                       input_shape=(X_train.shape[1], X_train.shape[2]
                                    )))
        model.add(Dense(1))
        #model.compile(loss=mean_absolute_error, optimizer=RAdamOptimizer(learning_rate=lr), metrics=[metric])
        model.compile(loss=mean_absolute_error, optimizer=Adam(lr=lr), metrics=[metric])
        history = model.fit(x=X_train, y=y_train,
                            epochs=500, batch_size=64,
                            validation_data=(X_valid, y_valid), verbose=1,
                            shuffle=False, callbacks=[earlyStopping, checkpoint])
        model.evaluate(X_test, y_test, verbose=0)

        
        y_test_pred = model.predict(X_test)
        y_test, y_test_pred = y_sc.inverse_transform(y_test), y_sc.inverse_transform(y_test_pred)
        y_test_pred[y_test_pred < 1] = 0
        score[ind, 3], score[ind, 4] = sklearn.metrics.mean_absolute_error(y_test, y_test_pred), np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))


        y_valid_pred = model.predict(X_valid)
        y_valid, y_valid_pred = y_sc.inverse_transform(y_valid), y_sc.inverse_transform(y_valid_pred)
        y_valid_pred[y_valid_pred < 1] = 0
        score[ind, 1], score[ind, 2] = sklearn.metrics.mean_absolute_error(y_valid, y_valid_pred), np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_valid_pred))
        score[ind, 0] = ind

        #history_dict = history.history
        #print(history_dict.keys())
        save_history(history, metric)
 
        start, end = 0, len(y_test)
        plt.figure(figsize=(16, 10))
        plt.plot(y_test_pred[start:end], linewidth=2, linestyle="-", color="r")
        plt.plot(y_test[start:end], linewidth=2, linestyle="-", color="b")
        plt.legend(["Predition", "Ground Truth"])
        plt.xlim(0, end - start)
        plt.ylim(-500, 2600)
        plt.grid(True)
        if not os.path.isdir(f'..//Plots{lr}'):
            os.makedirs(f'..//Plots{lr}')
        plt.savefig(f"..//Plots{lr}//PredictedStepTest_" + str(PREDICTED_STEP) + "_folds_" + str(ind + 1) + "_Original.png", dpi=50, bbox_inches="tight")
        plt.close("all")
    score = pd.DataFrame(score, columns=["fold", "validRMSE", "validMAE", "testRMSE", "testMAE"])
    
   


    
    #
    # plt.figure()
    # sns.distplot(trainData["target"].values, kde=True, bins=100)
    # sns.distplot(testData["target"].values, kde=True, color="green", bins=100)
    # plt.legend(["Train", "Test"])
    #
    # plt.figure()
    # sns.boxplot(data = [trainData["target"].values, testData["target"].values], color="green")
    
    
