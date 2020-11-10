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

warnings.filterwarnings('ignore')
np.random.seed(20201005)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1,
                    help='Turn GPU on(GPU number) or off(-1). Default is -1.')
parser.add_argument('--predictstep', type=int, default=10,
                    help='Choose the predicted step: 1, 10, 30, 50, 100. Default value is 10.')
parser.add_argument('--activation', type=str, default='mish',
                    help='Choose the activation function instead of mish: sigmoid, swish.')
parser.add_argument('--explore_hp', type=int, default='1',
                    help='Turn the parameter search on(1) or off(0). Default is 1.')
parser.add_argument('--hs', type=int,
                    help='Determine the hidden unit size of model. This option is valid only when explore_hp is 0.')
parser.add_argument('--lr', type=float,
                    help='Determine the learning rate of model. This option is valid only when explore_hp is 0.')
parser.add_argument('--bs', type=int,
                    help='Determine the batch size of model. This option is valid only when explore_hp is 0.')
args = parser.parse_args()

predicted_step = args.predictstep
rcr_activation = args.activation
param_search_switch = args.explore_hp
direct_hs = args.hs
direct_lr = args.lr
direct_bs = args.bs

# metric = 'mae'
if param_search_switch:
    hidden_sizes = [10, 14, 18, 22, 26, 30]
    lrs = [1e-4, 2e-4, 5e-4]  # [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    batch_sizes = [32, 64, 256]  # [32, 64, 128, 256, 512]
else:
    hidden_sizes = [direct_hs]
    lrs = [direct_lr]
    print(lrs)
    batch_sizes = [direct_bs]

os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
sns.set(style="ticks", font_scale=1.1, palette='deep', color_codes=True)
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, verbose=2)

PATH = f"..//Data//TrainedRes//sec{predicted_step}//"

swish = tf.keras.activations.swish
#######


def mish(x):
    """
    Mish is an activation function. Return mish(x) = x * tanh(ln(1+exp(x)).

    :param x: tensor object in TensorFlow
    :return: mish(x):
    """
    return x * tf.nn.tanh(tf.nn.softplus(x))
get_custom_objects().update({'mish': mish})


def plot_history(_history, result_dir):
    """
    Plot the history of loss and validation loss in some location.

    :param _history: model history, which is equal to model.fit.
    :param result_dir: location to save plots.
    """
    plt.figure()
    plt.plot(_history.history['loss'], marker='.')
    plt.plot(_history.history['val_loss'], marker='.')
    #plt.plot(history.history['accuracy'], marker='*')
    plt.title('Model')# Mean Absolute Error')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 0.1)
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(result_dir, dpi=500, bbox_inches="tight")
    plt.close()


def save_chkpt():
    """
    Save the checkpoint of model to some location.
    """
    with open(filepath + 'chkpt_best.pkl', 'wb') as f:
        pickle.dump(chkpt.best, f, protocol=pickle.HIGHEST_PROTOCOL)


def main_model(_X_train, _y_train,
               _X_valid, _y_valid,
               hs_info, rcr_act_info, lr_info, bs_info):
    """
    The core part of this model. Return LSTM model and model.fit.

    :param _X_train:
    :param _y_train:
    :param _X_valid:
    :param _y_valid:
    :param int hs_info: hidden unit size.
    :param rcr_act_info: recurrent activation function.
    :param float lr_info: learning rate.
    :param int bs_info: batch size.
    :return: model, history
    """
    _model = Sequential()
    _model.add(LSTM(hs_info,
                    recurrent_activation=rcr_act_info,
                    kernel_initializer="he_uniform",
                    recurrent_initializer="orthogonal",
                    return_sequences=False,
                    input_shape=(_X_train.shape[1], _X_train.shape[2])))
    _model.add(Dense(1))
    _model.compile(loss=mean_squared_error,
                   optimizer=Adam(lr=lr_info),
                   metrics=['mae'])
    _history = _model.fit(_X_train, _y_train,
                          epochs=500, batch_size=bs_info,
                          validation_data=(_X_valid, _y_valid), verbose=1,
                          shuffle=False,
                          callbacks=[earlyStopping, chkpt, save_chkpt_callback])

    return _model, _history
#######


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

                    chkpt = ModelCheckpoint(filepath=filepath + 'model.h5',
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
                    model, history = main_model(X_train, y_train,
                                                X_valid, y_valid,
                                                hidden_size, rcr_activation, lr, batch_size)
                    model.save(filepath + 'lastmodel.h5')
                    print('The trained model has been saved as "lastmodel.h5".')
                    del model
                    model = load_model(filepath + 'lastmodel.h5', custom_objects={'mish': mish})
                    model.evaluate(X_test, y_test, verbose=1)
                    # model.evaluate(X_test, y_test, verbose=0)

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
                    plot_history(history, filepath + f'error_pic{ind + 1}.png')
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
