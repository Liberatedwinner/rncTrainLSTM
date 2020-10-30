import numpy as np
import pandas as pd

np.random.seed(20201005)

#df = pd.read_excel('..//Data//trainKOR//180713_2.xlsx')
filenames = ['20180713.csv', '20180717.csv']
filename = '20180713.csv'
df = pd.read_csv(f'..//Data//metroKOR//{filename}')

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=10,
                    help='choose the predicted step: 1, 10, 30, 50, 100')
args = parser.parse_args()


df.drop(['번호', 'OP Mode', '편성번호', '열차길이', 'VOBC ＃1', 'VOBC ＃0',
         'Master Clock of VOBC', 'Train In Station.1',
         'Next Platform ID', 'Final Platform ID',
         'BC ＃1', 'BC ＃2', 'BC ＃3', 'BC ＃4',
         'Unnamed: 27', 'Unnamed: 28', 'BC ＃7', 'BC ＃0',
         'Train Room Temp ＃1', 'Train Outside Temp ＃1'
        ], axis=1, inplace=True)

df.rename(columns={'시간': 'time'}, inplace=True)
df.columns = df.columns.str.lower()

df['time'] = pd.to_datetime(filename[:-4] + df['time'].str.replace(':', ''))
#df['time'] = df['time'].str.replace(':', '')
#df['time'] = df['time'].astype('int64')

threewords = ['p/b', 'distance', 'line voltage', 'distance to target']

# 'p/b' (%)
# 'distance' (m)
# 'line voltage' (V)
# 'distance to target' (m)

for word in threewords:
    df[f'{word}'] = df[f'{word}'].str[:-3]
    df[f'{word}'] = df[f'{word}'].astype('int64')

df['mr pressure'] = df['mr pressure'].str[:-5] #(mpa?) (kpa?)
df['mr pressure'] = df['mr pressure'].str.replace('．', '.')
df['mr pressure'] = df['mr pressure'].astype('float64')

speedwords = ['target', 'permitted', 'actual', 'train']
for word in speedwords:
    df[f'{word} speed'] = df[f'{word} speed'].str[:-6]
    df[f'{word} speed'] = df[f'{word} speed'].astype('int64')


df = df[['time', 'p/b', 'motoring', 'braking', 'train speed'
        #'permitted speed', 'train speed',  # 'target speed', 'actual speed',
        #'distance to target', 'train in station'
        ]]

def lagging_features(data, name=None, laggingStep=[1, 2, 3]):
    assert name, "Invalid feature name!"

    for step in laggingStep:
        tmp = data[[name, "timeStep"]].copy()
        tmp.rename({name: "lagged_" + f'{name}_' + str(step)}, axis=1, inplace=True)
        tmp["timeStep"] += step
        data = pd.merge(data, tmp, on="timeStep", how="left")
    return data



def statistical_features(data, name=None, operation=["mean", "std"], timeRange=5):
    assert name, "Invalid feature name!"
    index = list(data.index)
    featureValues = data[name].values

    aggMean = []
    aggStd = []
    aggFirstLast = []
    for currInd in index:
        tmp = featureValues[max(0, currInd - timeRange):currInd]
        aggMean.append(np.nanmean(tmp))
        aggStd.append(np.nanstd(tmp))
        aggFirstLast.append(featureValues[currInd] - featureValues[max(0, currInd - timeRange)])

    data[name + "_lag_mean_" + str(timeRange)] = aggMean
    data[name + "_lag_std_" + str(timeRange)] = aggStd
    data[name + "_last_first_" + str(timeRange)] = aggFirstLast
    return data


def create_target(data, predictStep=None, targetName="speed"):
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
df.rename({"index":"timeStep"}, axis=1, inplace=True)

df = lagging_features(df, name="train speed", laggingStep=list(range(1, 11)) + [20, 30, 50, 80])
df = lagging_features(df, name="p/b", laggingStep=list(range(1, 6)) + [20, 60])

df['train speed_mult_0'] = df['train speed']
for k in range(1, 6):
    df[f'train speed_mult_{k}'] = df[f'train speed_mult_{k-1}'] * df[f'lagged_speed_{k}']


for k in [5, 10, 20]:
    df = statistical_features(df, name='train speed', timeRange=k)
    df = statistical_features(df, name='p/b', timeRange=k)
