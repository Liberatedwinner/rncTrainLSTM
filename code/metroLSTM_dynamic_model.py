# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from MetroLSTMCore import ModelCore
import numpy as np
import pandas as pd
import warnings
import argparse
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.utils import get_custom_objects
np.random.seed(20201005)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # GPU No.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
warnings.filterwarnings('ignore')

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=2)


hidden_sizes = [20]#[10, 14, 18, 22, 26, 30]
lrs = [1e-3]#[1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
batch_sizes = [256]#[32, 64, 128, 256, 512]
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
get_custom_objects().update({'mish': mish})
###############################################################################
def plot_history(history, result_dir):
    plt.figure()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.plot(history.history['accuracy'], marker='*')
    plt.title('Model')# Mean Absolute Error')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss', 'accuracy'], loc='upper right')
    plt.savefig(result_dir, dpi=500, bbox_inches="tight")
    plt.close()


def save_chkpt():
  with open(filepath + 'chkpt_best.pkl', 'wb') as f:
    pickle.dump(chkpt.best, f, protocol=pickle.HIGHEST_PROTOCOL)
###############################################################################


if __name__ == "__main__":
    mdc = ModelCore(PATH)
    trainData, testData = mdc.load_train_test_data()

    # Exclude
    testData["target"] = mdc.load_data('test_results.pkl')

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
                filepath = f'..//Plots-tanh_{rcr_activation}//{hidden_size}-{lr}-{batch_size}//'
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

                    chkpt = ModelCheckpoint(filepath=filepath + '//model.h5',
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True)

                    if os.path.exists(filepath + 'chkpt_best.pkl') and os.path.getsize(filepath + 'chkpt_best.pkl') > 0:
                        with open(filepath + 'chkpt_best.pkl', 'rb') as f:
                            best = pickle.load(f)
                            chkpt.best = best

                    save_chkpt_callback = LambdaCallback(
                        on_epoch_end=lambda epoch, logs: save_chkpt()
                    )

                    if rcr_activation == 'swish':
                        rcr_activation = swish
                    elif rcr_activation == 'sigmoid':
                        rcr_activation = 'sigmoid'
                    else:
                        rcr_activation = mish

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
                                  metrics=['mae', 'accuracy'])
                    history = model.fit(X_train, y_train,
                                        epochs=500, batch_size=batch_size,
                                        validation_data=(X_valid, y_valid), verbose=1,
                                        shuffle=False,
                                        callbacks=[earlyStopping, chkpt, save_chkpt_callback])
                    model.save(filepath + 'lastmodel.h5')
                    del model
                    model = load_model(filepath + 'lastmodel.h5', custom_objects={'mish': mish})
                    model.evaluate(X_test, y_test, verbose=1)
                    #model.evaluate(X_test, y_test, verbose=0)

                    y_valid = y_sc.inverse_transform(y_valid)
                    y_valid_pred = model.predict(X_valid)
                    y_valid_pred = y_sc.inverse_transform(y_valid_pred)
                    y_valid_pred[y_valid_pred < 1] = 0

                    y_test = y_sc.inverse_transform(y_test)
                    y_test_pred = model.predict(X_test)
                    y_test_pred = y_sc.inverse_transform(y_test_pred)
                    y_test_pred[y_test_pred < 1] = 0
                    
                    score[ind] = np.array([r2_score(y_test, y_test_pred),
                                           sklearn.metrics.mean_absolute_error(y_valid, y_valid_pred),
                                           np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_valid_pred)),
                                           sklearn.metrics.mean_absolute_error(y_test, y_test_pred),
                                           np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))
                                           ])
                    ModelCore(filepath).pred_drawing(y_test_pred, y_test, ind, predicted_step)
                    plot_history(history, filepath + f'errorpic{ind+1}.png')
                    print('The graph has been saved.\n')

                if rcr_activation == swish:
                    rcr_activation = 'swish'
                elif rcr_activation == 'sigmoid':
                    rcr_activation = 'sigmoid'
                else:
                    rcr_activation = 'mish'

                score = pd.DataFrame(score,
                                     columns=["R-square", "validMAE", "validRMSE", "testMAE", "testRMSE"])
                print(score)

                # saving the results
                filename = f'score-{hidden_size}-{lr}-{batch_size}'
                score.to_pickle(filepath + filename + '.pkl')
                print(f'The result has been saved as {filename}.pkl')
                score.to_csv(filepath + filename + '.csv')
                print(f'The result has been saved as {filename}.csv')

