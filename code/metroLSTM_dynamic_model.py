# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pandas as pd
import warnings
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
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from sklearn.metrics import r2_score
import argparse
import pickle
import tensorflow as tf

np.random.seed(20201005)

warnings.filterwarnings('ignore')
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=2)


hidden_sizes = [22, 26, 30]#[10, 14, 18, 22, 26, 30]
lrs = [1e-4, 1e-3, 2e-3]#[1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
batch_sizes = [32, 128, 512]#[32, 64, 128, 256, 512]
metric = 'mae'

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=10,
                    help='choose the predicted step: 1, 10, 30, 50, 100. Default value is 10.')
parser.add_argument('--activation', type=str, default='mish',
                    help='choose the activation function instead of mish: sigmoid, swish.')
args = parser.parse_args()
predicted_step = args.predictstep
rcr_activation = args.activation

PATH = f"..//Data//TrainedRes//sec{predicted_step}//"

swish = tf.keras.activations.swish


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))
###############################################################################
def load_train_test_data():
    ls = LoadSave(PATH + "train.pkl")
    trainData = ls.load_data()

    ls._fileName = PATH + "test.pkl"
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

    with open('result.txt', 'a') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write(f'{i}\t{loss[i]}\t{acc[i]}\t{val_loss[i]}\t{val_acc[i]}\n')

def save_chkpt():
  with open(filepath + '//chkpt_best.pkl', 'wb') as f:
    pickle.dump(chkpt.best, f, protocol=pickle.HIGHEST_PROTOCOL)
###############################################################################
if __name__ == "__main__":
    trainData, testData = load_train_test_data()

    # Exclude
    ls = LoadSave(PATH + "//test_results.pkl")
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
    for hidden_size in hidden_sizes:
        for lr in lrs:
            for batch_size in batch_sizes:
                score = np.zeros((numFolds, 5))
                filepath = f'..//Plots-tanh_{rcr_activation}//{hidden_size}-{lr}-{batch_size}'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)

                for ind, (train, valid) in enumerate(folds):
                    X_train = trainData.iloc[train].drop(["target"], axis=1).values
                    X_valid = trainData.iloc[valid].drop(["target"], axis=1).values

                    y_train = trainData.iloc[train]["target"].values.reshape(len(X_train), 1)
                    y_valid = trainData.iloc[valid]["target"].values.reshape(len(X_valid), 1)

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

                    chkpt = ModelCheckpoint(filepath=filepath + f'//model-{ind}.h5',
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True)

                    if os.path.exists(filepath + '//chkpt_best.pkl') and os.path.getsize(filepath + '//chkpt_best.pkl') > 0:
                        with open(filepath + '//chkpt_best.pkl', 'rb') as f:
                            best = pickle.load(f)
                            chkpt.best = best

                    save_chkpt_callback = LambdaCallback(
                        on_epoch_end=lambda epoch, logs: save_chkpt()
                    )

                    if rcr_activation == 'swish':
                        rcr_activation = swish
                    elif rcr_activation == 'sigmoid':
                        rcr_activation = 'sigmoid'

                    # Start training the model
                    model = Sequential()
                    model.add(LSTM(hidden_size,
                                   recurrent_activation=rcr_activation,
                                   kernel_initializer="he_uniform",
                                   recurrent_initializer="orthogonal",
                                   return_sequences=False,
                                   input_shape=(X_train.shape[1], X_train.shape[2])))
                    model.add(Dense(1))
                    model.compile(loss=mean_squared_error,
                                  optimizer=Adam(lr=lr),
                                  metrics=['mae'])
                    history = model.fit(X_train, y_train,
                                        epochs=500, batch_size=batch_size,
                                        validation_data=(X_valid, y_valid), verbose=1,
                                        shuffle=False,
                                        callbacks=[earlyStopping, chkpt, save_chkpt_callback])
                    model.evaluate(X_test, y_test, verbose=0)

                    y_valid_pred = model.predict(X_valid)
                    y_valid = y_sc.inverse_transform(y_valid)
                    y_valid_pred = y_sc.inverse_transform(y_valid_pred)
                    y_valid_pred[y_valid_pred < 1] = 0

                    y_test_pred = model.predict(X_test)
                    y_test = y_sc.inverse_transform(y_test)
                    y_test_pred = y_sc.inverse_transform(y_test_pred)
                    y_test_pred[y_test_pred < 1] = 0
                    
                    score[ind] = np.array([r2_score(y_test, y_test_pred),
                                           sklearn.metrics.mean_absolute_error(y_valid, y_valid_pred),
                                           np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_valid_pred)),
                                           sklearn.metrics.mean_absolute_error(y_test, y_test_pred),
                                           np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))
                                           ])
                    # score[ind, 0] = r2_score(y_test, y_test_pred)
                    # score[ind, 1] = sklearn.metrics.mean_absolute_error(y_valid, y_valid_pred)
                    # score[ind, 2] = np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_valid_pred))
                    # score[ind, 3] = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
                    # score[ind, 4] = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))

                    start, end = 0, len(y_test)
                    plt.figure(figsize=(16, 10))
                    plt.plot(y_test_pred[start:end], linewidth=2, linestyle="-", color="r")
                    plt.plot(y_test[start:end], linewidth=2, linestyle="-", color="b")
                    plt.legend(["Prediction", "Ground Truth"])
                    plt.xlim(1000, 2000) #plt.xlim(0, end - start)
                    plt.ylim(0, 120) #plt.ylim(-500, 2600)
                    plt.grid(True)
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    plt.savefig(filepath + f'//PredictedStepTest_{(predicted_step)}_folds_{ind + 1}.png',
                                dpi=50, bbox_inches="tight")
                    plt.close("all")

                if rcr_activation == swish:
                    rcr_activation = 'swish'
                elif rcr_activation == 'sigmoid':
                    rcr_activation = 'sigmoid'

                score = pd.DataFrame(score,
                                     columns=["R-square", "validMAE", "validRMSE", "testMAE", "testRMSE"])
                print(score)

                # saving the results
                filename = f'//score-{hidden_size}-{lr}-{batch_size}'
                score.to_pickle(filepath + filename + '.pkl')
                score.to_csv(filepath + filename + '.csv')
