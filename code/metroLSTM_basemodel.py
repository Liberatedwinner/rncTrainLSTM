# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
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
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.utils import get_custom_objects
from MetroLSTMCore import ModelCore
import MetroLSTMconfig
warnings.filterwarnings('ignore')
np.random.seed(20201005)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1,
                    help='Turn GPU on(GPU number) or off(-1). Default is -1.')
parser.add_argument('--predictstep', type=int, default=10,
                    help='Choose the predicted step: 1, 10, 30, 50, 100. Default value is 10.')
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
param_search_switch = args.explore_hp
direct_input_hs = args.hs
direct_input_lr = args.lr
direct_input_bs = args.bs

if param_search_switch:
    hidden_sizes = MetroLSTMconfig.MODEL_CONFIG['hidden_sizes']
    lrs = MetroLSTMconfig.MODEL_CONFIG['learning_rates']
    batch_sizes = MetroLSTMconfig.MODEL_CONFIG['batch_sizes']
else:
    hidden_sizes = [direct_input_hs]
    lrs = [direct_input_lr]
    batch_sizes = [direct_input_bs]

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
sns.set(style='ticks', font_scale=1.1, palette='deep', color_codes=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
PATH = MetroLSTMconfig.MODEL_CONFIG['path'] + f'sec{predicted_step}//'
#######


def mish(x):
    """
    Mish is an activation function. Return Mish(x) = x * tanh(ln(1+exp(x)).

    :param x: tensor object in TensorFlow
    :return: Mish(x)
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
    plt.plot(_history.history['loss'], marker='.', linewidth=1.5)
    plt.plot(_history.history['val_loss'], marker=',', linewidth=1.5)
    plt.title('Model loss and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 0.1)
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(result_dir, dpi=500, bbox_inches='tight')
    plt.close()


def save_chkpt():
    """
    Save the checkpoint of model to some location.
    """
    with open(filepath + 'chkpt_best.pkl', 'wb') as f:
        pickle.dump(chkpt.best, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_result(_filepath, _filename, _score):
    """
    Saving the result as '.pkl' and '.csv' files.

    :param _filepath:
    :param _filename:
    :param _score:
    """
    _score.to_pickle(_filepath + _filename + '.pkl')
    print(f'The result has been saved as {_filename}.pkl')
    _score.to_csv(_filepath + _filename + '.csv')
    print(f'The result has been saved as {_filename}.csv')


def drop_nan_data(_path):
    """
    This part is in order to get rid NaN value off from data.

    :param _path: file path where data is located.
    :return: traindata and testdata
    """
    mdc = ModelCore(_path)
    _trainData, _testData = mdc.load_train_test_data()

    # Exclude
    _testData['target'] = mdc.load_data('test_results.pkl')

    print(f'Train shape: {_trainData.shape}, Test shape: {_testData.shape} before dropping nan values.')
    _trainData.dropna(inplace=True)
    _testData.dropna(inplace=True)

    _trainData.drop('FLAG', axis=1, inplace=True)
    _testData.drop('FLAG', axis=1, inplace=True)
    print(f'Train shape: {_trainData.shape}, Test shape: {_testData.shape} After dropping nan values.')

    return _trainData, _testData


def prepare_to_parse_data(_traindata, _testdata, raw_train, raw_valid):
    """
    This part parses data into traindata and testdata.

    :param _traindata:
    :param _testdata:
    :param raw_train:
    :param raw_valid:
    :return: parsed data.
    """
    _X_train = _traindata.iloc[raw_train].drop(['target'], axis=1).values
    _X_valid = _traindata.iloc[raw_valid].drop(['target'], axis=1).values

    _y_train = _traindata.iloc[raw_train]['target'].values.reshape(len(_X_train), 1)
    _y_valid = _traindata.iloc[raw_valid]['target'].values.reshape(len(_X_valid), 1)

    return _X_train, _y_train, _X_valid, _y_valid


def main_model(_X_train, _y_train, _X_valid, _y_valid,
               _hidden_size, _recurrent_activation, _learning_rate, _batch_size):
    """
    The core part of this model. Return LSTM model and history = model.fit.

    :param _X_train:
    :param _y_train:
    :param _X_valid:
    :param _y_valid:
    :param int _hidden_size: hidden unit size.
    :param _recurrent_activation: recurrent activation function.
    :param float _learning_rate: learning rate.
    :param int _batch_size: batch size.
    :return: model, history
    """
    _model = Sequential()
    _model.add(LSTM(_hidden_size,
                    recurrent_activation=_recurrent_activation,
                    kernel_initializer='he_uniform',
                    recurrent_initializer='orthogonal',
                    return_sequences=False,
                    input_shape=(_X_train.shape[1], _X_train.shape[2])))
    _model.add(Dense(1))
    _model.compile(loss=mean_squared_error,
                   optimizer=Adam(lr=_learning_rate),
                   metrics=['mae'])
    _history = _model.fit(_X_train, _y_train,
                          epochs=500, batch_size=_batch_size,
                          validation_data=(_X_valid, _y_valid), verbose=1,
                          shuffle=False,
                          callbacks=[earlyStopping, chkpt, save_chkpt_callback])

    return _model, _history


def evaluate_model(_train_data, _test_data,
                   raw_train, raw_valid, _file_path,
                   _hidden_size, _learning_rate, _batch_size):
    """
    Evaluation part of the model.

    :param _train_data:
    :param _test_data:
    :param raw_train:
    :param raw_valid:
    :param _file_path:
    :param _hidden_size:
    :param _learning_rate:
    :param _batch_size:
    :return: y_valid, prediction of y_valid, y_test, prediction of y_test.
    """
    X_train, y_train, X_valid, y_valid = prepare_to_parse_data(_train_data, _test_data, raw_train, raw_valid)

    # Access the normalized data
    X_sc, y_sc = MinMaxScaler(), MinMaxScaler()

    X_train = X_sc.fit_transform(X_train)
    X_valid = X_sc.transform(X_valid)
    X_test = X_sc.transform(_test_data.drop(['target'], axis=1).values)

    y_train = y_sc.fit_transform(y_train)
    y_valid = y_sc.transform(y_valid)
    y_test = y_sc.transform(_test_data['target'].values.reshape(len(X_test), 1))

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    if os.path.exists(_file_path + 'chkpt_best.pkl') and os.path.getsize(_file_path + 'chkpt_best.pkl') > 0:
        with open(_file_path + 'chkpt_best.pkl', 'rb') as f:
            best = pickle.load(f)
            chkpt.best = best

    # Start training the model
    model, history = main_model(X_train, y_train, X_valid, y_valid,
                                _hidden_size, _learning_rate, _batch_size)
    model.save(_file_path + 'lastmodel.h5')
    print('The trained model has been saved as "lastmodel.h5".')
    del model
    model = load_model(_file_path + 'lastmodel.h5', custom_objects={'mish': mish})
    model.evaluate(X_test, y_test, verbose=1)

    y_valid = y_sc.inverse_transform(y_valid)
    y_valid_pred = model.predict(X_valid)
    y_valid_pred = y_sc.inverse_transform(y_valid_pred)
    y_valid_pred[y_valid_pred < 1] = 0

    y_test = y_sc.inverse_transform(y_test)
    y_test_pred = model.predict(X_test)
    y_test_pred = y_sc.inverse_transform(y_test_pred)
    y_test_pred[y_test_pred < 1] = 0

    return y_valid, y_valid_pred, y_test, y_test_pred, history


def post_training(func):
    """
    Decorator of a function 'trained_model_score'. This part is of saving data.

    :param func:
    """
    def wrapper(*args, **kwargs):
        # saving the results
        score = func(*args, **kwargs)
        filename = f'score-{hidden_size}-{lr}-{batch_size}'
        save_result(filepath, filename, score)

    return wrapper


@post_training
def trained_model_score(_filepath, _folds, _train_data, _test_data,
                        _hidden_size, _learning_rate, _batch_size):
    """
    This part is the model training block.

    :param _filepath: The path where file is located.
    :param _folds: split data with time series split method.
    :param _train_data:
    :param _test_data:
    :param _hidden_size: hidden unit size.
    :param _learning_rate: learning rate.
    :param _batch_size: batch size.
    :return: score, which is np.array.
    """
    _fold_number = MetroLSTMconfig.MODEL_CONFIG['fold_number']
    score = np.zeros((_fold_number, 5))
    for ind, (train, valid) in enumerate(_folds):
        y_valid, y_valid_pred, y_test, y_test_pred, history = evaluate_model(_train_data, _test_data,
                                                                             train, valid, _filepath,
                                                                             _hidden_size, _learning_rate, _batch_size)
        score[ind] = np.array([
            r2_score(y_test, y_test_pred),
            sklearn.metrics.mean_absolute_error(y_valid, y_valid_pred),
            np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_valid_pred)),
            sklearn.metrics.mean_absolute_error(y_test, y_test_pred),
            np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))
        ])

        print(f'R-square: {score[ind][0]}, test MAE: {score[ind][3]}')
        ModelCore(_filepath).pred_drawing(y_test_pred, y_test, ind, predicted_step)
        plot_history(history, _filepath + f'error_pic{ind + 1}.png')
        print('The metro speed-prediction graph has been saved.\n')

    score = pd.DataFrame(score,
                         columns=['R-square', 'validMAE', 'validRMSE', 'testMAE', 'testRMSE'])
    print(score)

    return score
#######


if __name__ == '__main__':
    trainData, testData = drop_nan_data(PATH)
    fold_number = MetroLSTMconfig.MODEL_CONFIG['fold_number']
    tscv = TimeSeriesSplit(n_splits=fold_number)
    folds = []
    for trainInd, validInd in tscv.split(trainData):
        folds.append([trainInd, validInd])
    save_chkpt_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: save_chkpt()
    )

    # Start the time series cross validation.
    for hidden_size in hidden_sizes:
        for lr in lrs:
            for batch_size in batch_sizes:
                filepath = f'..//Plots-base//{predicted_step}_{hidden_size}-{lr}-{batch_size}//'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                chkpt = ModelCheckpoint(filepath=filepath + 'model.h5',
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True)
                trained_model_score(filepath, folds, trainData, testData,
                                    hidden_size, lr, batch_size)
