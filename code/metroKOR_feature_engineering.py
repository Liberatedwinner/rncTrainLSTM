import numpy as np
import pandas as pd
import argparse

np.random.seed(20201005)

# df = pd.read_excel('..//Data//trainKOR//180713_2.xlsx')
filenames = ['20180713.csv', '20180717.csv']
filename = '20180713.csv'
df = pd.read_csv(f'..//Data//metroKOR//{filename}')

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=10,
                    help='choose the predicted step: 1, 10, 30, 50, 100')
args = parser.parse_args()
predicted_step = args.predictstep

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
df['time'] = pd.to_datetime(filename[:-4] + df['time'].str.replace(':', ''))

df['p/b'] = df['p/b'].str[:-3]
df['p/b'] = df['p/b'].astype('int64')

speedwords = ['permitted', 'actual', 'train']
for word in speedwords:
    df[f'{word} speed'] = df[f'{word} speed'].str[:-6]
    df[f'{word} speed'] = df[f'{word} speed'].astype('int64')

for i in range(1, 7):
    df[f'bc{i}'] = df[f'bc{i}'].str[:-5]
    df[f'bc{i}'] = df[f'bc{i}'].astype('float64')


def lagging_features(data, name=None, laggingStep=[1, 2, 3]):
    assert name, "Invalid feature name!"

    for step in laggingStep:
        tmpframe = data[[name, "timeStep"]].copy()
        tmpframe.rename({name: "lagged_" + f'{name}_' + str(step)}, axis=1, inplace=True)
        tmpframe["timeStep"] += step
        data = pd.merge(data, tmpframe, on="timeStep", how="left")
    return data


def statistical_features(data, name=None, operation=["mean", "std"], timeRange=5):
    assert name, "Invalid feature name!"
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

    data[name + "_lag_mean_" + str(timeRange)] = Means
    data[name + "_lag_std_" + str(timeRange)] = Stds
    data[name + "_diff_" + str(timeRange)] = Diffs
    return data


def create_target(data, predictStep=None, targetName="actual speed"):
    target = data[targetName].copy()
    newData = pd.DataFrame(None, columns=list(data.columns), dtype=np.float64)
    newData["target"] = None
    newData["timeFlag"] = None

    for step in predictStep:
        targetTmp = target[step:].copy()
        data["target"] = targetTmp.reset_index(drop=True)
        data["timeFlag"] = step
        newData = pd.concat([newData, data], axis=0, ignore_index=True)
    newData["timeFlag"] = newData["timeFlag"].astype(np.float64)
    return newData


df.reset_index(inplace=True, drop=True)
df.reset_index(inplace=True)
df.rename({"index": "timeStep"}, axis=1, inplace=True)

df = lagging_features(df, name="actual speed", laggingStep=list(range(1, 11)) + [20, 30, 50, 80])
df = lagging_features(df, name="p/b", laggingStep=list(range(1, 6)) + [20, 60])

df['train speed_mult_0'] = df['actual speed']
for k in range(1, 6):
    df[f'train speed_mult_{k}'] = df[f'train speed_mult_{k-1}'] * df[f'lagged_speed_{k}']


for k in [5, 10, 20]:
    df = statistical_features(df, name='actual speed', timeRange=k)
    df = statistical_features(df, name='p/b', timeRange=k)

df = create_target(df, predictStep=predicted_step, targetName="actual speed")

