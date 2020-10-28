import pandas as pd

#df = pd.read_excel('..//Data//trainKOR//180713_2.xlsx')
filename = '180713_2.csv'
df = pd.read_csv(f'..//Data//metroKOR//{filename}')
print(df.columns)

df.drop(['번호',
         'OP Mode',
         '편성번호',
         '열차길이',
         'VOBC ＃1',
         'VOBC ＃0',
         'Master Clock of VOBC',
         'Train In Station.1',
         'Next Platform ID',
         'Final Platform ID',
         'BC ＃1',
         'BC ＃2',
         'BC ＃3',
         'BC ＃4',
         'Unnamed: 27',
         'Unnamed: 28',
         'BC ＃7',
         'BC ＃0',
         'Train Room Temp ＃1',
         'Train Outside Temp ＃1'
        ], axis=1, inplace=True)

df.rename(columns={'시간': 'time'}, inplace=True)
df.columns = df.columns.str.lower()

df['time'] = pd.to_datetime('20' + filename[:-6] + df['time'].str.replace(':', ''))
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
