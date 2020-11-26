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
                    help='choose the predicted step: 1, 10, 30, 50, 100')
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

    df.rename({'BC ＃1': 'BC1', 'BC ＃2': 'BC2', 'BC ＃3': 'BC3',
               'BC ＃4': 'BC4', 'BC ＃7': 'BC5', 'BC ＃0': 'BC6'},
              axis=1, inplace=True)

    df.drop(['번호', 'OP Mode', '편성번호', '열차길이', 'VOBC ＃1', 'VOBC ＃0',
             'Master Clock of VOBC', 'Train In Station.1',
             'Next Platform ID', 'Final Platform ID',
             'Unnamed: 27', 'Unnamed: 28', 'Target Speed',
             'Train Room Temp ＃1', 'Train Outside Temp ＃1',
             'FWD', 'REV', 'Train In Station', 'Line Voltage',
             'DISTANCE', 'MR Pressure', 'Distance to Target'],
            axis=1, inplace=True)

    df.rename(columns={'시간': 'time'}, inplace=True)
    df.columns = df.columns.str.lower()
    df['time'] = pd.to_datetime(file_name[:-4] + df['time'].str.replace(':', ''))
    df['hour'] = df['time'].dt.hour
    df['dayOfWeek'] = df['time'].dt.dayofweek
    df['rest'] = df['dayOfWeek'] > 4 # 0-mon
    df['day'] = df['time'].dt.day
    df.drop(['time'], axis=1, inplace=True)

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


def feature_engineering(dataAll, predictStep=[10]):
    """
    Main function of this file. Indeed, this part proceeds the feature engineering.

    :param dataAll: dataframe.
    :param predictStep: array of timesteps.
    :return: newdata: feature-selected dataframe array.
    """
    FLAG = dataAll['FLAG'].unique()
    newData = []

    print('=======')
    for flag in FLAG:
        print('Running with the file {}:'.format(flag))
        data = dataAll[dataAll['FLAG'] == flag]

        for name in list(data.columns):
            if data[name].isnull().sum() <= 100:
                data[name].fillna(method='ffill', inplace=True)

        data.reset_index(inplace=True, drop=True)
        data.reset_index(inplace=True)
        data.rename({'index': 'timeStep'}, axis=1, inplace=True)

        print('Generating lagging features...')
        data = lagging_features(data,
                                name='actual speed',
                                laggingStep=list(range(1, 11)) +[20, 30, 50, 80])

        print('.')
        data = lagging_features(data,
                                name='permitted speed',
                                laggingStep=list(range(1, 11)) +[20, 30, 50, 80])

        print('.')
        data = lagging_features(data,
                                name='p/b',
                                laggingStep=[1, 3, 5, 20, 60])

        print('.')
        # data = lagging_features(data,
        #                         name='harmonic_bc',
        #                         laggingStep=[1, 3, 5, 20, 60])
        #
        # print('.')
        data['speed_mult_0'] = data['actual speed']
        for k in range(1, 6):
            data[f'speed_mult_{k}'] = data[f'speed_mult_{k-1}'] * data[f'lagged_actual speed_{k}']
        print('Completed.')

        print('Generating statistical features...')
        for k in [2, 5, 10, 20]:
            data = statistical_features(data,
                                        name='actual speed',
                                        timeRange=k)
            print('.')
            data = statistical_features(data,
                                        name='permitted speed',
                                        timeRange=k)
            print('.')
            data = statistical_features(data,
                                        name='p/b',
                                        timeRange=k)
            print('.')
            # data = statistical_features(data,
            #                             name='harmonic_bc',
            #                             timeRange=k)
            # print('.')
        print('Completed.')

        print('Marking the timestep flag to the target...')
        data = create_target(data,
                             predictStep=predictStep,
                             targetName='actual speed')
        data = data[~data['target'].isnull()]
        data.reset_index(inplace=True, drop=True)
        newData.append(data)
        print('Completed.')
    print('=======')

    return newData


def lagging_features(data,
                     name=None,
                     laggingStep=[1, 2, 3]):
    """
    This part makes lagged features.

    :param data: dataframe.
    :param name: feature name.
    :param laggingStep: array of timesteps.
    :return: data: dataframe with lagged features.
    """
    assert name, 'Invalid feature name.'

    for step in laggingStep:
        tmpframe = data[[name, 'timeStep']].copy()
        tmpframe.rename({name: 'lagged_' + f'{name}_' + str(step)}, axis=1, inplace=True)
        tmpframe['timeStep'] += step
        data = pd.merge(data, tmpframe, on='timeStep', how='left')

    return data


def statistical_features(data,
                         name=None,
                         timeRange=5):
    """
    This part makes statistical features.

    :param data: dataframe.
    :param name: feature name.
    :param timeRange: single timestep.
    :return: data: dataframe with statistical features.
    """
    assert name, 'Invalid feature name.'
    index = list(data.index)
    featureValues = data[name].values
    Means = []
    Stds = []
    Diffs = []

    for currInd in index:
        tmp = featureValues[max(0, currInd - timeRange):currInd]
        Means.append(np.nanmean(tmp))
        Stds.append(np.nanstd(tmp))
        Diffs.append(featureValues[currInd] - featureValues[max(0, currInd - timeRange)])

    data[name + '_lag_mean_' + str(timeRange)] = Means
    data[name + '_lag_std_' + str(timeRange)] = Stds
    data[name + '_diff_' + str(timeRange)] = Diffs

    return data


def create_target(data,
                  predictStep=None,
                  targetName='actual speed'):
    """
    This part marks the target feature.

    :param data: dataframe.
    :param predictStep: array of timesteps.
    :param targetName: Aim of feature prediction.
    :return: data with a marked target.
    """
    target = data[targetName].copy()
    newData = pd.DataFrame(None, columns=list(data.columns), dtype=np.float64)
    newData['target'] = None
    newData['timeFlag'] = None

    for step in predictStep:
        targetTmp = target[step:].copy()
        data['target'] = targetTmp.reset_index(drop=True)
        data['timeFlag'] = step
        newData = pd.concat([newData, data], axis=0, ignore_index=True)
    newData['timeFlag'] = newData['timeFlag'].astype(np.float64)

    return newData
#######


if __name__ == '__main__':
    filenames = ['20180717.csv', '20180713.csv']
    dfs = []
    for filename in filenames:
        dtf = preprocessing(filename)
        dfs.append(dtf)
    dfs = flag_setting(dfs)
    dataframe = data_concat(dfs)

    dataAll = feature_engineering(dataframe, predictStep=[predicted_step])
    print('\nMerging the data:')
    print('=======')
    shapeList = [len(df) for df in dataAll]
    print(f'Total shape is {sum(shapeList)}')
    newData = pd.DataFrame(None, columns=list(dataAll[0].columns))
    for idx, data in enumerate(dataAll):
        print(f'{idx}: {len(data)}.')
        newData = pd.concat([newData, data], axis=0, ignore_index=True)
    print('=======')

    dropList = ['timeStep', 'hour', 'dayOfWeek', 'rest', 'day', 'timeFlag', 'train speed', 'speed_mult_0']
    for i in range(1, 7):
       dropList.append(f'bc{i}')
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
