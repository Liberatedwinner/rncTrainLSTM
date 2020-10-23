import pandas as pd

# csv read
# 시간축에 따른 크기 그래프를 그려본다
# 어느 것이 속도랑 연관이 있을까 한번 재어봐야지
# 내 생각엔 P/B인데

#df = pd.read_excel('..//Data//trainKOR//180713_2.xlsx')
df = pd.read_csv('..//Data//trainKOR//180713_2.csv')
print(df.columns) # 한글로 된 거 영어로 바꿔야됨
df.info()
print(df.shape)
print(df.dtypes)
df.describe() # 통계값들
df.count() # 몇 개쯤 있나
df.corr() # 상관계수



