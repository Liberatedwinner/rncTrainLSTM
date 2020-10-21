#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import time
import gc
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings('ignore')

from WeaponLib import ReduceMemoryUsage
from WeaponLib import LoadSave
from WeaponLib import timefn
from WeaponLib import basic_feature_report

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)

parser = argparse.ArgumentParser()
parser.add_argument('--predictstep', type=int, default=10,
                    help='choose the predicted step: 1, 10, 30, 50, 100')
args = parser.parse_args()

###############################################################################
###############################################################################
@timefn
def preprocessing(nrows=None):
    def str_replace(t):
        return t.replace(":", "")
    
    fileName = ["20180711.csv", "20200807.csv"]
    path = "..//Data//"
    data = pd.read_csv(path + fileName[0], nrows=nrows)
    data["时间"] = pd.to_datetime(fileName[0][:-4] + data["时间"].apply(str_replace))
    data["FLAG"] = 0
    
    for ind, name in enumerate(fileName[1:2+1]):
        tmp = pd.read_csv(path + name, nrows=nrows)
        tmp["时间"] = pd.to_datetime(name[:-4] + tmp["时间"].apply(str_replace))
        tmp["FLAG"] = ind + 1
        data = pd.concat([data, tmp], ignore_index=True, axis=0)

    
    featureNames = ['时间', '系统', '序列号', '数据是否完整', '运行级别', 
                    '列车头部Link', '列车头部Offset', '列车头部方向', '列车速度', 'EBI',
                    '列车是否停稳', '当前站编号', '下一站编号', 'ATO主流程', 'ATO驾驶状态',
                    'ATO已激活', '牵引状态', '制动状态', 'ATO模拟量输出', '目标速度',
                    '目标距离', '当前坡度', 'PID控制器期望加速度', '记录停车阶段的误差', 'ATO输出模拟量', 
                    '列车参考速度', '列车重量', '网压', '网流', '牵引力',
                    '空气制动力', '电制动力', '转向架常用制动不可用数量', '转向架紧急制动不可用数量', 'TMS输入端口2偏移量9车辆施加牵引',
                    '车辆施加制动', '推荐速度', '牵引制动状态', '列车停车精度', 'ATO停车超限', '乘客满载率', 'FLAG']

    newName = ["time", "system", "seriesNumber", "dataCompleted", "runningLevel",
               "trainHead_link", "trainHead_offset", "heading", "speed", "EBI",
               "isStop", "currStationId", "nextStationId", "ATO_main", "ATO_status",
               "ATO_active", "draggingStatus", "brakeStatus", "ATO_sim", "targetSpeed",
               "targetDist", "slope", "PID_desireSpeed", "stopError", "ATO_simOut", 
               "referenceSpeed", "weight", "voltage", "current", "draggingForce",
               "airBrakeForce", "electricBrakeForce", "movingStickUnaviable", "movingStickUnaviableEmergnecy", "TMS",
               "vehBrake", "recommendedSpeed", "draggingBrakeStatus", "stopAccuracy", "ATO_stop", "fullHouseRate", 'FLAG']
    data.columns = newName
    indictor = data["FLAG"].values
    
    # Dropping some useless features
    data.drop(["seriesNumber", "system", "dataCompleted", "trainHead_link", "trainHead_offset",
                 "movingStickUnaviable", "movingStickUnaviableEmergnecy", "TMS", "stopAccuracy", "stopError"], axis=1, inplace=True)
    nameDict = {cName: eName for cName, eName in zip(featureNames, newName)}
    
    # Dealing with some of the missing values and error values
    data["runningLevel"].replace(255, np.nan, inplace=True)
    data["speed"].replace(65535, np.nan, inplace=True)
    data["EBI"].replace(65535, np.nan, inplace=True)
    data["currStationId"].replace(65535, np.nan, inplace=True)
    data["nextStationId"].replace(65535, np.nan, inplace=True)
    data["targetSpeed"].replace(65535, np.nan, inplace=True)
    data["targetDist"].replace(0, np.nan, inplace=True)
    data["recommendedSpeed"].replace(65535, np.nan, inplace=True)
    data["heading"].replace(["55", "55.0"], 55, inplace=True)
    data["isStop"].replace(["55", "55.0"], 55, inplace=True)
    data["isStop"].replace(["FF"], np.nan, inplace=True)
    data["isStop"].replace([55], True, inplace=True)
    data["isStop"].replace(["AA"], False, inplace=True)
    

    '''
    Numeric features : speed, ATO_sim, targetSpeed, ATO_status, draggingForce, targetDist
                       weight, airBrakeForce, recommendedSpeed
    '''
    data = data[["time", "ATO_sim", "speed", "slope", "draggingBrakeStatus"]]
    

    dataReport = basic_feature_report(data)
    featureNames = list(data.columns)
    for name in featureNames:
        if (dataReport["types"][dataReport["featureName"] == name] == "object").values[0]:
            lbl = LabelEncoder()
            data[name] = data[name].apply(str)
            data[name] = lbl.fit_transform(data[name].values)
    

    data["hour"] = data["time"].dt.hour
    data["dayOfWeek"] = data["time"].dt.dayofweek
    data["rest"] = data["dayOfWeek"] > 4 # 0-mon
    data["day"] = data["time"].dt.day
    data.drop(["time"], axis=1, inplace=True)
    

    data["FLAG"] = indictor
    mr = ReduceMemoryUsage(data)
    data = mr.reduce_memory_usage()
    
    # Save the data as the .pkl file
    ls = LoadSave("..//Data//AllInOne.pkl")
    ls.save_data(data=data)
    return data, nameDict

def load_data():
    ls = LoadSave("..//Data//AllInOne.pkl")
    data = ls.load_data()
    return data

###############################################################################
###############################################################################
def feature_engineering(dataAll, predictStep=[75]):
    FLAG = dataAll["FLAG"].unique()
    newData = []
    
    print("\n@MichaelYin : Start feature engineering:")
    print("=============================================================")
    print("Start time : {}".format(datetime.now()))
    for flag in FLAG:
        print("----------------------------------------------------")
        print("Start with the file {}:".format(flag))
        data = dataAll[dataAll["FLAG"] == flag]

        print("    Step 0. Forward fill nan, exclude the wrong times")
        data = data[(data["hour"] >= 5 ) & (data["hour"] < 23)]    # Only remain the non-stop data
        for name in list(data.columns):
            if data[name].isnull().sum() <= 100:
                data[name].fillna(method="ffill", inplace=True) # 결손 직전 값으로 이후 값을 채움
                
        data.reset_index(inplace=True, drop=True)
        data.reset_index(inplace=True)
        data.rename({"index":"timeStep"}, axis=1, inplace=True)     

        print("    Step 1. Create basic lagging features for each time step")
        data = lagging_features(data, name="speed", laggingStep=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 80])
        data = lagging_features(data, name="ATO_sim", laggingStep=[1, 3, 5, 20, 60])

        data["speed_cross_1"] = data["speed_lag_1"] * data["speed"]
        data["speed_cross_2"] = data["speed_lag_2"] * data["speed_lag_1"] * data["speed"]
        data["speed_cross_3"] = data["speed_lag_3"] * data["speed_lag_2"] * data["speed_lag_1"] * data["speed"]
        data["speed_cross_4"] = data["speed_lag_4"] * data["speed_lag_3"] * data["speed_lag_2"] * data["speed_lag_1"] * data["speed"]
        data["speed_cross_5"] = data["speed_lag_5"] * data["speed_lag_4"] * data["speed_lag_3"] * data["speed_lag_2"] * data["speed_lag_1"] * data["speed"]

        print("    Step 2. Create the statistical features for each time step")
        data = statistical_features(data, name="speed", timeRange=5)
        data = statistical_features(data, name="speed", timeRange=10)
        data = statistical_features(data, name="speed", timeRange=20)
        data = statistical_features(data, name="ATO_sim", timeRange=5)
        data = statistical_features(data, name="ATO_sim", timeRange=10)
        data = statistical_features(data, name="ATO_sim", timeRange=20)
        print("    Step 3. Creat the time step flag with the target")

        data = create_target(data, predictStep=predictStep, targetName="speed")
        
        data = data[~data["target"].isnull()]
        data.reset_index(inplace=True, drop=True)
        newData.append(data)
    print("End time : {}".format(datetime.now()))
    print("=============================================================")
    return newData
    
# Basic lagging step
def lagging_features(data, name=None, laggingStep=[1, 2, 3]):
    assert name, "Invalid feature name!"
    for step in laggingStep:
        tmp = data[[name, "timeStep"]].copy()
        tmp.rename({name: name + "_lag_" + str(step)}, axis=1, inplace=True)
        tmp["timeStep"] += step
        data = pd.merge(data, tmp, on="timeStep", how="left")
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

# Statistical features
def statistical_features(data, name=None, operation=["mean", "std"], timeRange=5):
    assert name, "Invalid feature name!"
    index = list(data.index)
    featureValues = data[name].values
    
    aggMean = []
    aggStd = []
    aggFirstLast = []
    for currInd in index:
        tmp = featureValues[max(0, currInd-timeRange):currInd]
        aggMean.append(np.nanmean(tmp))
        aggStd.append(np.nanstd(tmp))
        aggFirstLast.append(featureValues[currInd] - featureValues[max(0, currInd-timeRange)])
        
    data[name + "_lag_mean_" + str(timeRange)] = aggMean
    data[name + "_lag_std_" + str(timeRange)] = aggStd
    data[name + "_last_first_" + str(timeRange)] = aggFirstLast
    return data

###############################################################################
###############################################################################
if __name__ == "__main__":
    dataAll, nameDict = preprocessing(nrows=None)
    dataAll = load_data()

    PREDICTED_STEP = args.predictstep

    dataAll = feature_engineering(dataAll, predictStep=[PREDICTED_STEP])
    print("\nMerging the data:")
    print("=============================================================")
    print("Start time : {}".format(datetime.now()))
    shapeList = [len(df) for df in dataAll]
    print("Total shape is {}".format(sum(shapeList)))
    newData = pd.DataFrame(None, columns=list(dataAll[0].columns))
    for data in dataAll:
        print("Now is {}.".format(len(data)))
        newData = pd.concat([newData, data], axis=0, ignore_index=True)
    print("End time : {}".format(datetime.now()))
    print("=============================================================")
    
    # Drop the useless features
    dropList = ["timeStep", "hour", "dayOfWeek", "rest", "day", "timeFlag"]
    newData.drop(dropList, axis=1, inplace=True)


    # Saved all the data
    PATH = f"..//Data//TrainedRes//sec{PREDICTED_STEP}//"
    # if PREDICTED_STEP == 10:
    #     PATH = "..//Data//TrainedRes//sec10//"
    # elif PREDICTED_STEP == 30:
    #     PATH = "..//Data//TrainedRes//sec30//"
    # elif PREDICTED_STEP == 50:
    #     PATH = "..//Data//TrainedRes//sec50//"
    # else:
    #     PATH = "..//Data//TrainedRes//sec1//"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    ls = LoadSave()
    ls.save_data(data=newData[(newData["FLAG"] == 0)],
                     path=PATH + "Train.pkl")

    ls.save_data(path=PATH + "Test.pkl",
                     data=newData[(newData["FLAG"] == 1)].drop("target", axis=1))

    ls.save_data(path=PATH + "TestResults.pkl",
                     data=newData[(newData["FLAG"] == 1)]["target"].values)
