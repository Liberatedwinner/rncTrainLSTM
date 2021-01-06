import numpy as np
import pandas as pd
import os
import argparse
import pickle
import warnings
from scipy.stats import hmean
from MetroLSTMCore import ModelCore

warnings.filterwarnings('ignore')
np.random.seed(20201005)

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=10,
                    help='choose the predicted step: for example, 1, 10, 30, 50, etc.')
args = parser.parse_args()
predicted_step = args.predictstep
#######


def preprocessing(file_name):
    """
    Preprocess the data. You need to pre-convert '.xlsx' file to '.csv' file.

    :param file_name: 'file_name.csv'
    :return: df: dataframe which has features below: p/b, motoring, braking, {permitted, actual, train} speed, bc#s
    """
    df = pd.read_csv(f'..//Data//metroKOR//{file_name}')

    df.rename(
        {'BC ＃1': 'BC1', 'BC ＃2': 'BC2', 'BC ＃3': 'BC3',
         'BC ＃4': 'BC4', 'BC ＃7': 'BC5', 'BC ＃0': 'BC6'},
         axis=1,
         inplace=True
    )

    df.drop(
        ['번호', '시간', 'OP Mode', '편성번호',
         '열차길이', 'VOBC ＃1', 'VOBC ＃0',
         'Master Clock of VOBC', 'Train In Station.1',
         'Next Platform ID', 'Final Platform ID',
         'Unnamed: 27', 'Unnamed: 28', 'Target Speed',
         'Train Room Temp ＃1', 'Train Outside Temp ＃1',
         'FWD', 'REV', 'Train In Station', 'Line Voltage',
         'DISTANCE', 'MR Pressure', 'Distance to Target'],
         axis=1,
        inplace=True
    )
    df.columns = df.columns.str.lower()

    df['p/b'] = df['p/b'].str[:-3]
    df['p/b'] = df['p/b'].astype('int64')

    speedwords = ['permitted', 'actual', 'train']
    for word in speedwords:
        df[f'{word} speed'] = df[f'{word} speed'].str[:-6]
        df[f'{word} speed'] = df[f'{word} speed'].astype('int64')

    for i in range(1, 7):
        df[f'bc{i}'] = df[f'bc{i}'].str[:-5]
        df[f'bc{i}'] = df[f'bc{i}'].astype('float64')
    df['harmonic_bc'] = hmean(df.loc[:, 'bc1':'bc6'], axis=1)

    return df


def flag_setting(df_lst):
    """
    This part is for flag setting.

    :param df_lst: array of dataframe.
    :return: df_lst: df_lst with 'flag'.
    """
    for ind, df in enumerate(df_lst):
        df['FLAG'] = ind

    return df_lst


def data_concat(df_lst):
    """
    This part is for 'pd.concat' of dataframes.
    As you can see, please import pandas as pd.

    :param df_lst: array of dataframe.
    :return: concatd_data: Concatenated dataframe.
    """
    concatd_data = pd.concat(df_lst, ignore_index=True, axis=0)
    with open('..//Data//concatd_data.pkl', 'wb') as f:
        pickle.dump(concatd_data, f)

    return concatd_data


def feature_engineering(data_all, predict_step=[10]):
    """
    Main function of this file. Indeed, this part proceeds the feature engineering.

    :param data_all: dataframe.
    :param predict_step: array of timesteps.
    :return: new_data: feature-selected dataframe array.
    """
    FLAG = data_all['FLAG'].unique()
    new_data = []

    print('=======')
    for flag in FLAG:
        print('Running with the file {}:'.format(flag))
        _data = data_all[data_all['FLAG'] == flag]

        for name in list(_data.columns):
            if _data[name].isnull().sum() <= 100:
                _data[name].fillna(method='ffill', inplace=True)

        _data.reset_index(inplace=True, drop=True)
        _data.reset_index(inplace=True)
        _data.rename({'index': 'timeStep'}, axis=1, inplace=True)

        print('Generating lagging features...')
        _data = lagging_features(
            _data,
            name='actual speed',
            lagging_step=list(range(1, 11)) + [20, 30, 50, 80]
        )
        _data = lagging_features(
            _data,
            name='permitted speed',
            lagging_step=list(range(1, 11)) + [20, 30, 50, 80]
        )
        _data = lagging_features(
            _data,
            name='p/b',
            lagging_step=[1, 3, 5, 20, 60]
        )
        _data['speed_mult_0'] = _data['actual speed']
        for k in range(1, 6):
            _data[f'speed_mult_{k}'] = _data[f'speed_mult_{k - 1}'] * _data[f'lagged_actual speed_{k}']
        print('Completed.')

        print('Generating statistical features...')
        for k in [2, 5, 10, 20]:
            _data = statistical_features(
                _data,
                name='actual speed',
                time_range=k
            )
            _data = statistical_features(
                _data,
                name='permitted speed',
                time_range=k
            )
            _data = statistical_features(
                _data,
                name='p/b',
                time_range=k
            )
        print('Completed.')

        print('Marking the timestep flag to the target...')
        _data = create_target(
            _data,
            predict_step=predict_step,
            target_name='actual speed')
        _data = _data[~_data['target'].isnull()]
        _data.reset_index(inplace=True, drop=True)
        new_data.append(_data)
        print('Completed.')
    print('=======')

    return new_data


def lagging_features(_data,
                     name=None,
                     lagging_step=[1, 2, 3]):
    """
    This part makes lagged features.

    :param _data: dataframe.
    :param name: feature name.
    :param lagging_step: array of timesteps.
    :return: data: dataframe with lagged features.
    """
    assert name, 'Invalid feature name.'

    for step in lagging_step:
        tmpframe = _data[[name, 'timeStep']].copy()
        tmpframe.rename(
            {name: 'lagged_' + f'{name}_' + str(step)},
            axis=1,
            inplace=True
        )
        tmpframe['timeStep'] += step
        _data = pd.merge(_data, tmpframe, on='timeStep', how='left')

    return _data


def statistical_features(_data, name=None, time_range=5):
    """
    This part makes statistical features.

    :param _data: dataframe.
    :param name: feature name.
    :param time_range: single timestep.
    :return: data: dataframe with statistical features.
    """
    assert name, 'Invalid feature name.'
    index = list(_data.index)
    feature_values = _data[name].values
    means = []
    stds = []
    diffs = []

    for currInd in index:
        tmp = feature_values[max(0, currInd - time_range):currInd]
        means.append(np.nanmean(tmp))
        stds.append(np.nanstd(tmp))
        diffs.append(feature_values[currInd] - feature_values[max(0, currInd - time_range)])
    _data[name + '_lag_mean_' + str(time_range)] = means
    _data[name + '_lag_std_' + str(time_range)] = stds
    _data[name + '_diff_' + str(time_range)] = diffs

    return _data


def create_target(_data, predict_step=None, target_name='actual speed'):
    """
    This part marks the target feature.

    :param _data: dataframe.
    :param predict_step: array of timesteps.
    :param target_name: Aim of feature prediction.
    :return: data with a marked target.
    """
    target = _data[target_name].copy()
    new_data = pd.DataFrame(None, columns=list(_data.columns), dtype=np.float64)
    new_data['target'] = None
    new_data['timeFlag'] = None

    for step in predict_step:
        target_tmp = target[step:].copy()
        _data['target'] = target_tmp.reset_index(drop=True)
        _data['timeFlag'] = step
        new_data = pd.concat([new_data, _data], axis=0, ignore_index=True)
    new_data['timeFlag'] = new_data['timeFlag'].astype(np.float64)

    return new_data
#######


if __name__ == '__main__':
    filenames = ['20180717.csv', '20180713.csv']
    dfs = []
    for filename in filenames:
        dtf = preprocessing(filename)
        dfs.append(dtf)
    dfs = flag_setting(dfs)
    dataframe = data_concat(dfs)

    dataAll = feature_engineering(dataframe, predict_step=[predicted_step])
    print('\nMerging the data:')
    print('=======')
    shapeList = [len(df) for df in dataAll]
    print(f'Total shape is {sum(shapeList)}')
    newData = pd.DataFrame(None, columns=list(dataAll[0].columns))
    for idx, data in enumerate(dataAll):
        print(f'{idx}: {len(data)}.')
        newData = pd.concat([newData, data], axis=0, ignore_index=True)
    print('=======')

    dropList = ['timeStep', 'timeFlag', 'train speed', 'speed_mult_0']
    for j in range(1, 7):
        dropList.append(f'bc{j}')
    newData.drop(dropList, axis=1, inplace=True)

    # Save all the data
    print('Preparing to save data...')
    PATH = f'..//Data//TrainedRes//sec{predicted_step}//'

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    train_data = newData[(newData['FLAG'] == 0)]
    test_data = newData[(newData['FLAG'] == 1)].drop('target', axis=1)
    test_result = newData[(newData['FLAG'] == 1)]['target'].values

    mdc = ModelCore(PATH)
    mdc.save_data('train.pkl', data=train_data)
    mdc.save_data('test.pkl', data=test_data)
    mdc.save_data('test_results.pkl', data=test_result)
