import pandas as pd

#df = pd.read_excel('..//Data//trainKOR//180713_2.xlsx')
df = pd.read_csv('..//Data//metroKOR//180713_2.csv')

df.drop(['번호',
         'OP Mode',
         '편성번호',
         '열차길이',
         'VOBC ＃1',
         'VOBC ＃0',
         'Unnamed: 27',
         'Unnamed: 28'
        ], axis=1, inplace=True)

df.rename(columns={'시간': 'time'}, inplace=True)
df.columns = df.columns.str.lower()

df['p/b'] = df['p/b'].str[:-3]
df['p/b'] = df['p/b'].astype('int64')

df['time'] = df['time'].str.replace(':', '')
df['time'] = df['time'].astype('int64')

df['distance'] = df['distance'].str[:-3]
df['distance'] = df['distance'].astype('int64')

speedwords = ['target', 'permitted', 'actual', 'train']
for word in speedwords:
    df[f'{word} speed'] = df[f'{word} speed'].str[:-6]
    df[f'{word} speed'] = df[f'{word} speed'].astype('int64')
