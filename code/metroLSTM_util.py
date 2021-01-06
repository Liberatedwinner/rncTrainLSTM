import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import get_custom_objects
from MetroLSTMCore import ModelCore
import MetroLSTMconfig

earlyStopping = MetroLSTMconfig.MODEL_CONFIG['early_stopping']


def mish(x):
    """
    Return Mish(x) = x * tanh(ln(1+exp(x)).

    :param x: tensor object in TensorFlow.
    :return: Mish(x)
    """
    return x * tf.nn.tanh(tf.nn.softplus(x))


get_custom_objects().update({'mish': mish})


def plot_history(_history, result_dir):
    """
    Plot the history of loss and validation loss.

    :param _history: model history, which is equal to model.fit.
    :param result_dir: location to save plots.
    """
    plt.figure()
    plt.plot(_history.history['loss'], marker='.', linewidth=1.5)
    plt.plot(_history.history['val_loss'], marker='+', linewidth=1.5)
    plt.title('Model loss and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 0.1)
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(result_dir, dpi=500, bbox_inches='tight')
    plt.close()


def save_chkpt(filepath_, chkpt_):
    """
    Save the checkpoint of model to some location.
    """
    with open(filepath_ + 'chkpt_best.pkl', 'wb') as f:
        pickle.dump(chkpt_.best, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_result(filepath_, filename_, score_):
    """
    Saving the result as '.pkl' and '.csv' files.

    :param filepath_:
    :param filename_:
    :param score_:
    """
    score_.to_pickle(filepath_ + filename_ + '.pkl')
    print(f'The result has been saved as {filename_}.pkl')
    score_.to_csv(filepath_ + filename_ + '.csv')
    print(f'The result has been saved as {filename_}.csv')


def drop_nan_data(_path):
    """
    This part is in order to get rid NaN value off from data.

    :param _path: file path where data is located.
    :return: traindata and testdata.
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


def prepare_to_parse_data(traindata_, raw_train, raw_valid):
    """
    This part parses data into train_data and valid_data.

    :param traindata_:
    :param raw_train:
    :param raw_valid:
    :return: parsed data.
    """
    _x_train = traindata_.iloc[raw_train].drop(['target'], axis=1).values
    _x_valid = traindata_.iloc[raw_valid].drop(['target'], axis=1).values

    _y_train = traindata_.iloc[raw_train]['target'].values.reshape(len(_x_train), 1)
    _y_valid = traindata_.iloc[raw_valid]['target'].values.reshape(len(_x_valid), 1)

    return _x_train, _y_train, _x_valid, _y_valid


def main_model(x_train_, y_train_, x_valid_, y_valid_,
               hidden_size_, recurrent_activation_, learning_rate_, batch_size_):
    """
    The crucial part of this model. Return LSTM model and history = model.fit.

    :param x_train_:
    :param y_train_:
    :param x_valid_:
    :param y_valid_:
    :param int hidden_size_: hidden unit size.
    :param recurrent_activation_: recurrent activation function.
    :param float learning_rate_: learning rate.
    :param int batch_size_: batch size.
    :return: model, history.
    """
    _model = Sequential()
    _model.add(LSTM(
        hidden_size_,
        recurrent_activation=recurrent_activation_,
        kernel_initializer='he_uniform',
        recurrent_initializer='orthogonal',
        return_sequences=False,
        recurrent_dropout=0.1,
        input_shape=(x_train_.shape[1], x_train_.shape[2]))
    )
    _model.add(Dense(1))
    _model.compile(
        loss=mean_squared_error,
        optimizer=Adam(lr=learning_rate_),
        metrics=['mae']
    )
    _history = _model.fit(
        x_train_, y_train_,
        epochs=500, batch_size=batch_size_,
        validation_data=(x_valid_, y_valid_), verbose=1,
        shuffle=False,
        callbacks=[earlyStopping, chkpt, save_chkpt_callback]
    )

    return _model, _history


def evaluate_model(train_data_, test_data_,
                   raw_train, raw_valid, filepath_,
                   hidden_size_, learning_rate_, batch_size_):
    """
    Evaluation part of the model.

    :param train_data_:
    :param test_data_:
    :param raw_train:
    :param raw_valid:
    :param filepath_:
    :param hidden_size_:
    :param learning_rate_:
    :param batch_size_:
    :return: y_valid, prediction of y_valid, y_test, prediction of y_test.
    """
    x_train, y_train, x_valid, y_valid = prepare_to_parse_data(train_data_, raw_train, raw_valid)

    # Access the normalized data
    x_sc, y_sc = MinMaxScaler(), MinMaxScaler()

    x_train = x_sc.fit_transform(x_train)
    x_valid = x_sc.transform(x_valid)
    x_test = x_sc.transform(test_data_.drop(['target'], axis=1).values)

    y_train = y_sc.fit_transform(y_train)
    y_valid = y_sc.transform(y_valid)
    y_test = y_sc.transform(test_data_['target'].values.reshape(len(x_test), 1))

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    if os.path.exists(filepath_ + 'chkpt_best.pkl') and os.path.getsize(filepath_ + 'chkpt_best.pkl') > 0:
        with open(filepath_ + 'chkpt_best.pkl', 'rb') as f:
            best = pickle.load(f)
            chkpt.best = best

    # Start training the model
    model, history = main_model(
        x_train, y_train, x_valid, y_valid,
        hidden_size_, recurrent_activation,
        learning_rate_, batch_size_
    )

    # Saving and evaluate the model
    model.save(filepath_ + 'saved_model.h5')
    print('The trained model has been saved as "saved_model.h5"')
    del model
    model = load_model(filepath_ + 'saved_model.h5', custom_objects={'mish': mish})
    print('Start the evaluation...')
    model.evaluate(x_test, y_test, verbose=1)

    y_valid = y_sc.inverse_transform(y_valid)
    y_valid_pred = model.predict(x_valid)
    y_valid_pred = y_sc.inverse_transform(y_valid_pred)
    y_valid_pred[y_valid_pred < 1] = 0

    y_test = y_sc.inverse_transform(y_test)
    y_test_pred = model.predict(x_test)
    y_test_pred = y_sc.inverse_transform(y_test_pred)
    y_test_pred[y_test_pred < 1] = 0

    return y_valid, y_valid_pred, y_test, y_test_pred, history


def post_training(func):
    """
    Decorator of a function 'trained_model_score'. This part is for saving data.

    :param func:
    """
    def wrapper(*argss, **kwargs):
        # saving the results
        filepath = FILE_PATH
        score = func(*argss, **kwargs)
        filename = f'score-{hs}-{lr}-{bs}'
        save_result(filepath, filename, score)

    return wrapper


@post_training
def trained_model_score(filepath_, folds_, train_data_, test_data_,
                        hidden_size_, learning_rate_, batch_size_):
    """
    This part is the model training block.

    :param filepath_: The path where file is located.
    :param folds_: split data with time series split method.
    :param train_data_:
    :param test_data_:
    :param hidden_size_: hidden unit size.
    :param learning_rate_: learning rate.
    :param batch_size_: batch size.
    :return: score, which is np.array.
    """
    _fold_number = MetroLSTMconfig.MODEL_CONFIG['fold_number']
    score = np.zeros((_fold_number, 5))
    for ind, (train, valid) in enumerate(folds_):
        y_valid, y_valid_pred, y_test, y_test_pred, history = evaluate_model(
            train_data_, test_data_,
            train, valid, filepath_,
            hidden_size_, learning_rate_, batch_size_
        )
        score[ind] = np.array([
            r2_score(y_test, y_test_pred),
            mean_absolute_error(y_valid, y_valid_pred),
            np.sqrt(mean_squared_error(y_valid, y_valid_pred)),
            mean_absolute_error(y_test, y_test_pred),
            np.sqrt(mean_squared_error(y_test, y_test_pred))
        ])

        print(f'R-square: {score[ind][0]}, test MAE: {score[ind][3]}')
        ModelCore(filepath_).pred_drawing(y_test_pred, y_test, ind, predicted_step)
        plot_history(history, filepath_ + f'error_pic{ind + 1}.png')
        print('The metro speed-prediction graph has been saved.\n')

    score = pd.DataFrame(
        score,
        columns=['R-square', 'validMAE', 'validRMSE', 'testMAE', 'testRMSE'])

    print(score)

    return score