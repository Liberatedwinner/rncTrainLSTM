# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:04:42 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import seaborn as sns
import time
import pickle
from functools import wraps
import pandas as pd
from pandas import DataFrame
import gc

from sklearn.metrics import roc_curve, auc, precision_recall_curve, recall_score, make_scorer, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import random
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def save_data(data, fileName=None):
    assert fileName, "Invalid file name !"
    print("-------------------------------------")
    print("Save file to {}".format(fileName))
    f = open(fileName, 'wb')
    pickle.dump(data, f)
    f.close()
    print("Save successed !")
    print("-------------------------------------")

def load_data(fileName=None):
    assert fileName, "Invalid file name !"
    print("-------------------------------------")
    print("Load file from {}".format(fileName))
    f = open(fileName, 'rb')
    data = pickle.load(f)
    f.close()
    print("Load successed !")
    print("-------------------------------------")
    return data

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fn.__name__ + " took {:5f}".format(end-start) + " seconds")
        return result
    return measure_time

@timefn
def replace_inf_with_nan(data):
    featureNameList = list(data.columns)
    for name in featureNameList:
        data[name].replace([np.inf, -np.inf], np.nan, inplace=True)
    return data

@timefn
def basic_feature_report(data):
    basicReport = data.isnull().sum()
    sampleNums = len(data)
    basicReport = pd.DataFrame(basicReport, columns=["missingNums"])
    basicReport["missingPrecent"] = basicReport["missingNums"]/sampleNums
    basicReport["nuniqueValues"] = data.nunique(dropna=False).values
    basicReport["types"] = data.dtypes.values
    basicReport.reset_index(inplace=True)
    basicReport.rename(columns={"index":"featureName"}, inplace=True)
    dataDescribe = data.describe([0.01, 0.5, 0.99, 0.995, 0.9995]).transpose()
    dataDescribe.reset_index(inplace=True)
    dataDescribe.rename(columns={"index":"featureName"}, inplace=True)
    basicReport = pd.merge(basicReport, dataDescribe, on='featureName', how='left')
    return basicReport

def drop_most_empty_features(data=None, precent=None):
    assert precent, "@MichaelYin: Invalid missing precent !"
    dataReport = basic_feature_report(data)
    featureName = list(dataReport["featureName"][dataReport["missingPrecent"] >= precent].values)
    data.drop(featureName, axis=1, inplace=True)
    return data, featureName

class LoadSave(object):
    def __init__(self, fileName=None):
        self._fileName = fileName
    
    def save_data(self, data=None, path=None):
        if path is None:
            assert self._fileName != None, "Invaild file path !"
            self.__save_data(data)
        else:
            self._fileName = path
            self.__save_data(data)
    
    def load_data(self, path=None):
        if path is None:
            assert self._fileName != None, "Invaild file path !"
            return self.__load_data()
        else:
            self._fileName = path    
            return self.__load_data()
        
    def __save_data(self, data=None):
        print("--------------Start saving--------------")
        print("SAVING DATA TO {}.".format(self._fileName))
        f = open(self._fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("SAVING SUCESSED !")
        print("---------------------------------------\n")
        
    def __load_data(self):
        assert self._fileName != None, "Invaild file path !"
        print("--------------Start loading--------------")
        print("LOADING DATA FROM {}.".format(self._fileName))
        f = open(self._fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("LOADING SUCCESSED !")
        print("----------------------------------------\n")
        return data

'''
-------------------------------------------------------------------------------
Author: Michael Yin
Date: 2018/12/26
Modified: 2018/12/26
Mail: zhuoyin94@163.com
Title: Loading and saving data to database.
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''
class LoadSaveFromDatabase(object):
    def __init__(self, databasePath=None, sheetName=None):
        self.databasePath = databasePath
        self.sheetName = sheetName
    
    @timefn
    def load_data(self, databasePath=None, sheetName=None):
        if databasePath is None:
            databasePath = self.databasePath
        if sheetName is None:
            sheetName = self.sheetName
        
        assert databasePath, "Invalid data base path !"
        assert sheetName, "Invalid path or sheet name!"
        
        print("\n--------------------------------------------------")
        print("LOADING FROM DATABASE PATH: {}.".format(databasePath))
        print("SHEET NAME :{}.".format(sheetName))
        sqlConnection = sqlite3.connect(databasePath)
        data = pd.read_sql_query("SELECT * FROM " + sheetName, sqlConnection)
        print("\nSUCCESSED !")
        print("----------------------------------------------------")
        sqlConnection.commit()
        sqlConnection.close()
        return data
    
    @timefn
    def save_data(self, data, databasePath=None, sheetName=None, if_exists='replace'):
        if databasePath is None:
            databasePath = self.databasePath
        if sheetName is None:
            sheetName = self.sheetName
        
        assert databasePath, "Invalid data base path !"
        assert sheetName, "Invalid path or sheet name!"
        
        print("\n--------------------------------------------------")
        print("SAVING TO DATABASE PATH: {}.".format(databasePath))
        print("SHEET NAME :{}.".format(sheetName))
        sqlConnection = sqlite3.connect(databasePath)
        data.to_sql(sheetName, sqlConnection, if_exists=if_exists, index=False)
        print("\nSUCCESSED !")
        print("----------------------------------------------------")
        sqlConnection.commit()
        sqlConnection.close()
###############################################################################
###############################################################################
# Time series feature selection
class TimeSeriesFeatureSelection(object):
    def __init__(self, dataTrain=None, dataTrainRes=None, verbose=False, randomState=2018, evalMethod='mse'):
        self.dataTrain = dataTrain
        self.dataTrainRes = dataTrainRes
        self.verbose = verbose
        self.importance = pd.DataFrame(None, columns=["featureName"])
        self.importance["featureName"] = list(self.dataTrain.columns)
        self.report = {}
        
    def create_cv_fold(self, timeFolds=[29, 30, 31, 32, 33]):
        self.splitFolds = []
        for fold in timeFolds:
            trainId = np.array(self.dataTrain[self.dataTrain["date_block_num"] < fold].index)
            trainResId = np.array(self.dataTrain[self.dataTrain["date_block_num"] == fold].index)
            self.splitFolds.append([trainId, trainResId])
    
    ###########################################################################
    # Set parameters for each regressor
    def set_random_forest_param(self, params=None):
        self.rfParams = params
        
    def set_lightgbm_param(self, params=None):
        self.lgbParams = params
    
    def set_xgboost_param(self, params=None):
        self.xgbParams = params
    
    ###########################################################################
    def train_xgboost(self, params):
        assert params, "Invalid parameters !"
        assert self.splitFolds, "Invalid folds !"
        
        print("\nXgboost cross validation feature importance:")
        n_folds = len(self.splitFolds)
        importances = 0
        reportTmp = np.zeros((n_folds, 4))
        for fold, (trainId, validationId) in enumerate(self.splitFolds):
            print("\n-------------------------------------------")
            print("Start fold {}, total is {} folds:".format(fold+1, n_folds))
            X_train, y_train = self.dataTrain.iloc[trainId], self.dataTrainRes.iloc[trainId]
            X_valid, y_valid = self.dataTrain.iloc[validationId], self.dataTrainRes.iloc[validationId]
            clf = xgb.XGBRegressor(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values),
                                                              (X_valid.values, y_valid.values)],
                                                              early_stopping_rounds=30,
                                                              eval_metric="rmse")
            score = clf.evals_result()
            scoreKeys = list(score.keys())
            
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["rmse"][clf.best_iteration]
            reportTmp[fold, 2] = score[scoreKeys[1]]["rmse"][clf.best_iteration]
            reportTmp[fold, 3] = clf.best_iteration
            importances += clf.feature_importances_
        report = DataFrame(reportTmp, columns=["fold", "loss", "validRes", 'bestIteration'])
        self.importance["xgbImportances"] = importances/n_folds
        self.report["XGBoost"] = report
        print("-------------------------------------------\n")
        
    def train_lightgbm(self, params=None):
        assert params, "Invalid parameters !"
        assert self.splitFolds, "Invalid folds !"
        
        params["boosting_type"] = 'gbdt'
        params["nthread"] = -1
        params['subsample_freq'] = 1
        n_folds = len(self.splitFolds)
        importance = 0
        reportTmp = np.zeros((n_folds, 4))
        print("\nLightGBM cross validation feature importance:")
        # Start K-fold validation
        for fold, (trainId, validationId) in enumerate(self.splitFolds):
            print("\n-------------------------------------------")
            print("Start fold {}, total is {} folds:".format(fold+1, n_folds))
            X_train, y_train = self.dataTrain.iloc[trainId], self.dataTrainRes.iloc[trainId]
            X_valid, y_valid = self.dataTrain.iloc[validationId], self.dataTrainRes.iloc[validationId]
            clf = lgb.LGBMRegressor(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
                                                              early_stopping_rounds=30, verbose=self.verbose)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["rmse"][clf.best_iteration_-1]
            reportTmp[fold, 2] = score[scoreKeys[1]]["rmse"][clf.best_iteration_-1]
            reportTmp[fold, 3] = clf.best_iteration_
            importance += clf.feature_importances_
            gc.collect()
        report = DataFrame(reportTmp, columns=["fold", "loss", "validRes", 'bestIteration'])
        self.importance["lgbImportances"] = importance/n_folds
        self.report["LightGBM"] = report
        print("-------------------------------------------\n")
    
    def train_random_forest(self, params):
        assert params, "Invalid parameters !"
        assert self.splitFolds, "Invalid folds !"
        
        n_folds = len(self.splitFolds)
        importance = 0
        reportTmp = np.zeros((n_folds, 4))
        print("\nLightGBM Random Forest cross validation feature importance:")
        # Start K-fold validation
        for fold, (trainId, validationId) in enumerate(self.splitFolds):
            print("\n-------------------------------------------")
            print("Start fold {}, total is {} folds:".format(fold+1, n_folds))
            X_train, y_train = self.dataTrain.iloc[trainId], self.dataTrainRes.iloc[trainId]
            X_valid, y_valid = self.dataTrain.iloc[validationId], self.dataTrainRes.iloc[validationId]
            clf = lgb.LGBMRegressor(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
                                                              early_stopping_rounds=30, verbose=self.verbose)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["rmse"][clf.best_iteration_-1]
            reportTmp[fold, 2] = score[scoreKeys[1]]["rmse"][clf.best_iteration_-1]
            reportTmp[fold, 3] = clf.best_iteration_
            importance += clf.feature_importances_
            gc.collect()
        report = DataFrame(reportTmp, columns=["fold", "loss", "validRes", 'bestIteration'])
        self.importance["rfImportances"] = importance/n_folds
        self.report["RandomForest"] = report
        print("-------------------------------------------\n")
    
    def feature_importances(self, lgbImportance=True, xgbImportance=False, rfImportance=False, timeFolds=[29, 30, 31, 32, 33]):
        self.create_cv_fold(timeFolds=timeFolds)
        if lgbImportance == True:
            self.train_lightgbm(self.lgbParams)
        if xgbImportance == True:
            self.train_xgboost(self.xgbParams)
        if rfImportance == True:
            self.train_random_forest(self.rfParams)
        return self.importance, self.report

###############################################################################
###############################################################################
'''
-------------------------------------------------------------------------------
Author: Michael Yin
Date: 2018/12/21
Modified: 2018/12/21
Mail: zhuoyin94@163.com
Title: LightGBM Random search of parameters for the Predicting Future Sales Competition.
-------------------------------------------------------------------------------
self.__init__(self, dataTrain=None, dataTrainRes=None, n_estimators=None, paramGrid=None, verbose=False, evalMethod='rmse'):
    Both DataTrain and dataTrainRes are pandas DataFrame. The paramGrid contains
    all possible parameters. n_estimators stands for how many different parameters
    you want to evaluate. The evalMethod is set to be the lower, the better.

self.get_score(self):
    Return the feature importance and cross validation results of each fold.

self.create_cv_fold(self, timeFolds=[33]):
    Using the time series cross validation method to create the training and validation
    fold index.

self.random_search_cv(self, random_state=2018, folds=[32, 33]):
    Evaluate the random search result. User should call this function to evaluate
    the parameter results.
    
self.__cross_validation(self, params=None, random_state=0):
    Private method for the cross validation. Return the average importance of each fold,
    the cross validation results and the best regressor.

self.create_submission(self, testData, save=True):
    Create the submission file, and save the result to the local.
-------------------------------------------------------------------------------
'''
class LgbTimeSeriesRandomSearch():
    def __init__(self, dataTrain=None, dataTrainRes=None, n_estimators=None, paramGrid=None, verbose=False, evalMethod='rmse'):
        self._dataTrain, self._dataTrainRes = dataTrain, dataTrainRes
        self._verbose = verbose
        self._evalMetod = evalMethod
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._paramGrid = paramGrid
        
        self._estimators = n_estimators
        self._report = {}
        self._bestParam = {}
        
    def get_score(self):
        return self._importance, self._report, self._bestParam
    
    def create_cv_fold(self, timeFolds=[33]):
        self._splitFolds = []
        for fold in timeFolds:
            trainId = np.array(self._dataTrain[self._dataTrain["date_block_num"] < fold].index)
            trainResId = np.array(self._dataTrain[self._dataTrain["date_block_num"] == fold].index)
            self._splitFolds.append([trainId, trainResId])
    
    def random_search_cv(self, random_state=2018, folds=[32, 33]):
        print("-------------------------------------------")
        bestEvalResMean = 1
        self.create_cv_fold(timeFolds=folds)
        
        for paramInd in range(self._estimators):
            print("\nNow is the {} estimator, total has {} estimators:".format(paramInd, self._estimators))
            cvRes = {}
            cvRes["param"] = {k: random.sample(list(v), 1)[0] for k, v in self._paramGrid.items()}
            importance, cvReport, regressor = self.__cross_validation(params=cvRes["param"], random_state=random_state)
            cvRes["cvReport"] = cvReport
            cvRes["validResMean"] = cvReport["validRes"].mean()
            cvRes["validResStd"] = cvReport["validRes"].std()
            cvRes["regressor"] = regressor
            
            self._report[paramInd] = cvRes
            self._importance["param_" + str(paramInd)] = importance
            
            # Update the best parameter and save the validating results
            if -cvRes["validResMean"] > -bestEvalResMean:
                self._bestParam["param"] = cvRes["param"]
                self._bestParam["paramInd"] = paramInd
                self._bestParam["cvReport"] = cvReport
                self._bestParam["validResMean"] = cvRes["validResMean"]
                self._bestParam["validResStd"] = cvRes["validResStd"]
                self._bestParam["bestRegressor"] = cvRes["regressor"]
                bestEvalResMean = cvRes["validResMean"]
    
    def __cross_validation(self, params=None, random_state=0):
        assert params, "Invalid parameters !"
        assert self._splitFolds, "Invalid folds !"
        
        params["objective"] = self._evalMetod
        params["boosting_type"] = 'gbdt'
        params["nthread"] = -1
        params['subsample_freq'] = 1
        params["silent"] = self._verbose
        n_folds = len(self._splitFolds)
        importance = 0
        reportTmp = np.zeros((n_folds, 4))
        print("\nLightGBM cross validation:")
        # Start K-fold validation
        for fold, (trainId, validationId) in enumerate(self._splitFolds):
            print("-------------------------------------------")
            print("Start fold {}, total is {} folds:".format(fold+1, n_folds))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainRes.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainRes.iloc[validationId]
            clf = lgb.LGBMRegressor(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
                                                              early_stopping_rounds=50, verbose=self._verbose)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["rmse"][clf.best_iteration_-1]
            reportTmp[fold, 2] = score[scoreKeys[1]]["rmse"][clf.best_iteration_-1]
            reportTmp[fold, 3] = clf.best_iteration_
            importance += clf.feature_importances_
            gc.collect()
        report = DataFrame(reportTmp, columns=["fold", "loss", "validRes", 'bestIteration'])
        importance = importance/n_folds
        print("-------------------------------------------\n")
        return importance, report, clf
    
    def create_submission(self, testData, save=True):
        regressor = self._bestParam["bestRegressor"]
        bestRes = self._bestParam["validResMean"]
        bestIteration = regressor.best_iteration_
        testData["item_cnt_month"] = regressor.predict(testData.drop(['item_cnt_month', 'ID'], axis=1).values, num_iteration=bestIteration)
        testData["item_cnt_month"] = testData["item_cnt_month"].clip(0, 20)
        testData["ID"] = testData["ID"].astype(int)
        
        if save == True:
            # Step 1: save the model.
            self.save_data(regressor, "..//TrainedModel//lgb_model_cv" + str(bestRes) + ".pkl")
            # Step_2: save the predictions.
            self.save_data(testData[["ID", "item_cnt_month"]], "..//SubmissionData//lgb_submission_cv"  + str(bestRes) + ".pkl")
        return testData[["ID", "item_cnt_month"]]
    
    def save_data(self, data, fileName=None):
        self.__save_data(data, fileName)
        
    def __save_data(self, data, fileName=None):
        assert fileName, "Invalid file name !"
        print("-------------------------------------")
        print("Save data to {}".format(fileName))
        f = open(fileName, 'wb')
        pickle.dump(data, f)
        f.close()
        print("Save successed !")
        print("-------------------------------------")
###############################################################################
###############################################################################
'''
-------------------------------------------------------------------------------
Author: Michael Yin
Date: 2018/12/21
Modified: 2018/12/21
Mail: zhuoyin94@163.com
Title: XGBoost random search of parameters for the Predicting Future Sales Competition.
-------------------------------------------------------------------------------
self.__init__(self, dataTrain=None, dataTrainRes=None, n_estimators=None, paramGrid=None, verbose=False, evalMethod='rmse'):
    DataTrain and dataTrainRes both are pandas data frame. The paramGrid contains
    all possible parameters. n_estimators stands for how many different parameters
    you want to evaluate. The evalMethod is set to be the lower, the better.

self.get_score(self):
    Return the feature importance and cross validation results of each fold.

self.create_cv_fold(self, timeFolds=[33]):
    Using the time series cross validation method to create the training and validation
    fold index.

self.random_search_cv(self, random_state=2018, folds=[32, 33]):
    Evaluate the random search result. User should call this function to evaluate
    the parameter results.
    
self.__cross_validation(self, params=None, random_state=0):
    Private method for the cross validation. Return the average importance of each fold,
    the cross validation results and the best regressor.

self.create_submission(self, testData, save=True):
    Create the submission file, and save to the local.
-------------------------------------------------------------------------------
'''
class XgbTimeSeriesRandomSearch():
    def __init__(self, dataTrain=None, dataTrainRes=None, n_estimators=None, paramGrid=None, verbose=False, evalMethod='rmse'):
        self._dataTrain, self._dataTrainRes = dataTrain, dataTrainRes
        self._verbose = verbose
        self._evalMetod = evalMethod
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._paramGrid = paramGrid
        
        self._estimators = n_estimators
        self._report = {}
        self._bestParam = {}
        
    def get_score(self):
        return self._importance, self._report, self._bestParam
    
    def create_cv_fold(self, timeFolds=[33]):
        self._splitFolds = []
        for fold in timeFolds:
            trainId = np.array(self._dataTrain[self._dataTrain["date_block_num"] < fold].index)
            trainResId = np.array(self._dataTrain[self._dataTrain["date_block_num"] == fold].index)
            self._splitFolds.append([trainId, trainResId])
    
    def random_search_cv(self, random_state=2018, folds=[32, 33]):
        print("-------------------------------------------")
        bestEvalResMean = 10
        self.create_cv_fold(timeFolds=folds)
        
        for paramInd in range(self._estimators):
            print("\nNow is the {} estimator, total has {} estimators:".format(paramInd, self._estimators))
            cvRes = {}
            cvRes["param"] = {k: random.sample(list(v), 1)[0] for k, v in self._paramGrid.items()}
            importance, cvReport, regressor = self.__cross_validation(params=cvRes["param"], random_state=random_state)
            cvRes["cvReport"] = cvReport
            cvRes["validResMean"] = cvReport["validRes"].mean()
            cvRes["validResStd"] = cvReport["validRes"].std()
            cvRes["regressor"] = regressor
            
            self._report[paramInd] = cvRes
            self._importance["param_" + str(paramInd)] = importance
            
            # Update the best parameter and save the validating results
            if -cvRes["validResMean"] > -bestEvalResMean:
                self._bestParam["param"] = cvRes["param"]
                self._bestParam["paramInd"] = paramInd
                self._bestParam["cvReport"] = cvReport
                self._bestParam["validResMean"] = cvRes["validResMean"]
                self._bestParam["validResStd"] = cvRes["validResStd"]
                self._bestParam["bestRegressor"] = cvRes["regressor"]
                bestEvalResMean = cvRes["validResMean"]
            gc.collect()
                
    def __cross_validation(self, params=None, random_state=100):
        assert params, "Invalid parameters !"
        assert self._splitFolds, "Invalid folds !"
        
        print("XGBoost cross validation:")
        importance = 0
        params["booster"] = 'gbtree'
        params["nthread"] = -1
        params["silent"] = self._verbose
        n_folds = len(self._splitFolds)
        importances = 0
        reportTmp = np.zeros((n_folds, 4))
        for fold, (trainId, validationId) in enumerate(self._splitFolds):
            print("-------------------------------------------")
            print("Start fold {}, total is {} folds:".format(fold+1, n_folds))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainRes.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainRes.iloc[validationId]
            clf = xgb.XGBRegressor(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values),
                                                              (X_valid.values, y_valid.values)],
                                                              early_stopping_rounds=50,
                                                              eval_metric="rmse")
            score = clf.evals_result()
            scoreKeys = list(score.keys())
            
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["rmse"][clf.best_iteration]
            reportTmp[fold, 2] = score[scoreKeys[1]]["rmse"][clf.best_iteration]
            reportTmp[fold, 3] = clf.best_iteration
            importances += clf.feature_importances_
            
        report = DataFrame(reportTmp, columns=["fold", "loss", "validRes", 'bestIteration'])
        return importance/n_folds, report, clf
        print("-------------------------------------------\n")

    def create_submission(self, testData, save=True):
        regressor = self._bestParam["bestRegressor"]
        bestRes = self._bestParam["validResMean"]
        bestIteration = regressor.best_iteration
        testData["item_cnt_month"] = regressor.predict(testData.drop(['item_cnt_month', 'ID'], axis=1).values, ntree_limit=bestIteration)
        testData["item_cnt_month"] = testData["item_cnt_month"].clip(0, 20)
        testData["ID"] = testData["ID"].astype(int)
        if save == True:
            # Step 1: save the model.
            self.save_data(regressor, "..//TrainedModel//lgb_model_cv" + str(bestRes) + ".pkl")
            # Step_2: save the predictions.
            self.save_data(testData[["ID", "item_cnt_month"]], "..//SubmissionData//lgb_submission_cv"  + str(bestRes) + ".pkl")
            
        return testData[["ID", "item_cnt_month"]]
    
    def save_data(self, data, fileName=None):
        self.__save_data(data, fileName)
        
    def __save_data(self, data, fileName=None):
        assert fileName, "Invalid file name !"
        print("-------------------------------------")
        print("Save data to {}".format(fileName))
        f = open(fileName, 'wb')
        pickle.dump(data, f)
        f.close()
        print("Save successed !")
        print("-------------------------------------")

###############################################################################
###############################################################################
class FeatureSelection():
    def __init__(self, dataTrain=None, dataTrainLabel=None, stratified=True, verbose=False):
        self._dataTrain = dataTrain
        self._dataTrainLabel = dataTrainLabel
        self._verbose = verbose
        self._stratified = stratified
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._report = DataFrame(None, columns=["fold", "trainScore", "validScore"])
    
    @timefn    
    def lgb_feature_importance(self, n_folds=3, shuffle=True, random_state=0):
        self.__lgb_feature_importance(n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    
    def __lgb_feature_importance(self, n_folds=None, shuffle=True, random_state=0):
        print("\nLightGBM cross validation feature importance:")
        print("-------------------------------------------")
        importances = 0
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 3))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("\n-------------------------------------------")
            print("Start fold {}".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            params = {
                    'n_estimators': 5000,
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'nthread': -1,
                    'learning_rate': 0.02,
                    'num_leaves': 20,
                    'subsample': 0.8715623,
                    'subsample_feq': 1,
                    'colsample_bytree':0.9497036,
                    'reg_alpha': 0.04154,
                    'reg_lambda': 0.0735294,
                    'silent': self._verbose
                    }
            self._params = params
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=20)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["binary_logloss"][-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            importances += clf.feature_importances_
        self._importance["lgb"] = importances/n_folds
        save_data(clf, "..//TrainedModel//LightGBM.pkl")
        print("-------------------------------------------")
    
    @timefn    
    def xgb_feature_importance(self, n_folds=3, shuffle=True, random_state=0):
        self.__xgb_feature_importance(n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    
    def __xgb_feature_importance(self, n_folds=None, shuffle=True, random_state=0):
        print("\nXgboost cross validation feature importance:")
        importances = 0
        print("-------------------------------------------")
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 3))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("\n-------------------------------------------")
            print("Start fold {}".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            params = {'max_depth':6,
                      "booster": "gbtree",
                      'n_estimators': 2000,
                      'learning_rate': 0.05,
                      'subsample': 0.85,
                      'colsample_bylevel': 0.632,
                      'colsample_bytree': 0.7,
                      'silent': self._verbose,
                      'objective':'binary:logistic',
                      'eval_metric':'auc',
                      'seed': random_state,
                      'nthread': -1,
                      'missing': np.nan}
            self._params = params
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=20)
            score = clf.evals_result()
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["auc"][-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            importances += clf.feature_importances_
        self._importance["xgb"] = importances/n_folds
        print("-------------------------------------------")
        
###############################################################################
###############################################################################
'''
Create the lightgbm and xgboost submission files.
'''
def create_lgb_submission(data, save=True):
    X_train = data[data["date_block_num"] < 33].drop(['item_cnt_month', 'ID'], axis=1)
    y_train = data[data["date_block_num"] < 33]["item_cnt_month"]
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'ID'], axis=1)
    y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month', "ID"], axis=1)
    submission = data[data.date_block_num == 34][["ID", "item_cnt_month"]].copy()
    
    importances = pd.DataFrame(None)
    importances["featureName"] = list(data.drop(['item_cnt_month', 'ID'], axis=1).columns)
    ###########################################################################
    # Training LightGBM
    lgbParams = {
            'n_estimators': 4000,
            'objective': 'rmse',
            'boosting_type': 'gbdt',
            'nthread': -1,
            'learning_rate': 0.063,
            'num_leaves': 50, # important
            'max_depth': 6, # important
            # 'min_split_gain': 0,
            'min_child_weight': 300,
            # 'min_child_samples': 20,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree':0.85,
            'reg_alpha': 12.56154,
            'reg_lambda': 12.5735294,
            'silent': 1
            }
    reg = lgb.LGBMRegressor(**lgbParams)
    reg.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
                                                      early_stopping_rounds=50)
    
    importances["importances"] = reg.feature_importances_
    y_test = reg.predict(X_test.values, num_iteration=reg.best_iteration_)
    ###########################################################################
    submission["item_cnt_month"] = y_test
    submission["ID"] = submission["ID"].astype(int)
    submission["item_cnt_month"] = submission["item_cnt_month"].clip(0, 20)
    
    scoreKeys = list(reg.evals_result_)
    score = reg.evals_result_[scoreKeys[1]]["rmse"][reg.best_iteration_-1]
    ###########################################################################
    # Save predictions
    if save == True:
        ls = LoadSave()
        ls._fileName = "..//TrainedModel//LightGBM_Model_" + str(score)[:8] + ".pkl"
        ls.save_data(reg)
        
        submission.to_csv("..//SubmissionData//LightGBM_Res_(" + str(score)[:8] + ").csv", index=False)
    return submission, importances


def create_xgb_submission(data, save=True):
    X_train = data[data["date_block_num"] < 33].drop(['item_cnt_month', 'ID'], axis=1)
    y_train = data[data["date_block_num"] < 33]["item_cnt_month"]
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'ID'], axis=1)
    y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month', "ID"], axis=1)
    submission = data[data.date_block_num == 34][["ID", "item_cnt_month"]].copy()
    
    importances = pd.DataFrame(None)
    importances["featureName"] = list(data.drop(['item_cnt_month', 'ID'], axis=1).columns)
    ###########################################################################
    # Training LightGBM
    xgbParams = {
            "max_depth":6,
            "n_estimators":4000,
            "min_child_weight":310, 
            "colsample_bytree":0.8, 
            "colsample_bylevel":0.8,
            "subsample":0.8, 
            "learning_rate":0.075,
            "n_jobs":-1,
            "reg_alpha":10.01,
            "reg_lambda":15.02,
            "booster":"gbtree"
            }
    reg = xgb.XGBRegressor(**xgbParams)
    reg.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values),
                                                      (X_valid.values, y_valid.values)],
                                                      early_stopping_rounds=50,
                                                      eval_metric="rmse")
    y_test = reg.predict(X_test.values, ntree_limit=reg.best_iteration)
    importances["importances"] = reg.feature_importances_
    ###########################################################################
    submission["item_cnt_month"] = y_test
    submission["ID"] = submission["ID"].astype(int)
    submission["item_cnt_month"] = submission["item_cnt_month"].clip(0, 20)
    
    scoreKeys = list(reg.evals_result().keys())
    score = reg.evals_result()[scoreKeys[1]]["rmse"][reg.best_iteration]
    ###########################################################################
    # Save predictions
    if save == True:
        ls = LoadSave()
        ls._fileName = "..//TrainedModel//XGBoost_Model_" + str(score)[:8] + ".pkl"
        ls.save_data(reg)
        
        submission.to_csv("..//SubmissionData//XGBoost_Res_(" + str(score)[:8] + ").csv", index=False)
    return submission, importances


###############################################################################
###############################################################################
# Reduce training data memory
class ReduceMemoryUsage():
    def __init__(self, data=None, verbose=True):
        self._data = data
        self._verbose = verbose
    
    def types_report(self, data):
        dataTypes = data.dtypes.values
        basicReport = pd.DataFrame(dataTypes, columns=["types"])
        basicReport["featureName"] = list(data.columns)
        return basicReport
    
    @timefn
    def reduce_memory_usage(self):
        self.__reduce_memory()
        return self._data
    
    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memoryStart = self._data.memory_usage(deep=True).sum() / 1024**2
        if self._verbose == True:
            print("@Memory usage of data is {:5f} MB.".format(memoryStart))
        self._types = self.types_report(self._data)
        for ind, name in enumerate(self._types["featureName"].values):
            featureType = str(self._types[self._types["featureName"] == name]["types"])
            if featureType != "object":
                featureMin = self._data[name].min()
                featureMax = self._data[name].max()
                if "int" in featureType:
                    # np.iinfo for reference:
                    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                    
                    # numpy data types reference:
                    # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                    if featureMin > np.iinfo(np.int8).min and featureMax < np.iinfo(np.int8).max:
                        self._data[name] = self._data[name].astype(np.int8)
                    elif featureMin > np.iinfo(np.int16).min and featureMax < np.iinfo(np.int16).max:
                        self._data[name] = self._data[name].astype(np.int16)
                    elif featureMin > np.iinfo(np.int32).min and featureMax < np.iinfo(np.int32).max:
                        self._data[name] = self._data[name].astype(np.int32)
                    elif featureMin > np.iinfo(np.int64).min and featureMax < np.iinfo(np.int64).max:
                        self._data[name] = self._data[name].astype(np.int64)
                else:
                    if featureMin > np.finfo(np.float32).min and featureMax < np.finfo(np.float32).max:
                        self._data[name] = self._data[name].astype(np.float32)
                    else:
                        self._data[name] = self._data[name].astype(np.float64)
            if self._verbose == True:
                print("Processed {} feature, total is {}.".format(ind+1, len(self._types)))
        memoryEnd = self._data.memory_usage(deep=True).sum() / 1024**2
        if self._verbose == True:
            print("@Memory usage after optimization: {:5f} MB.".format(memoryEnd))
            print("@Decreased by {}%".format(100 * (memoryStart - memoryEnd) / memoryStart))
        print("-------------------------------------------")
        
###############################################################################
# XGBoost Cross Validation
class RandomSearchCVXGBoost():
    def __init__(self, dataTrain=None, dataTrainLabel=None, n_estimators=None, stratified=True, paramGrid=None, verbose=False, randomState=2018):
        self._dataTrain, self._dataTest, self._dataTrainLabel, self._dataTestLabel = train_test_split(dataTrain, dataTrainLabel,
                                                                                                      test_size=0.15)
        self._stratified = stratified
        self._verbose = verbose
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._paramGrid = paramGrid
        self._report = {}
        self._estimators = n_estimators
        
    def get_score(self):
        return self._importance, self._report
    
    def random_search_cv(self, n_folds=3, random_state=0):
        print("-------------------------------------------")
        self._bestParam = {}
        bestAucMean = 0
        
        for paramInd in range(self._estimators):
            print("\nNow is {}:".format(paramInd))
            cvRes = {}
            cvRes["param"] = {k: random.sample(list(v), 1)[0] for k, v in self._paramGrid.items()}
            importance, cvReport = self.__cross_validation(n_folds=n_folds, random_state=random_state, params=cvRes["param"])
            cvRes["cvReport"] = cvReport
            cvRes["validAucMean"] = cvReport["validAuc"].mean()
            cvRes["validAucStd"] = cvReport["validAuc"].std()
            cvRes["testAucMean"] = cvReport["testAuc"].mean()
            cvRes["testAucStd"] = cvReport["testAuc"].std()
            self._report[paramInd] = cvRes
            self._importance["param_" + str(paramInd)] = importance
            if cvRes["validAucMean"] > bestAucMean:
                self._bestParam["param"] = cvRes["param"]
                self._bestParam["paramInd"] = paramInd
                self._bestParam["cvReport"] = cvReport
                self._bestParam["validAucMean"] = cvRes["validAucMean"]
                self._bestParam["validAucStd"] = cvRes["validAucStd"]
                self._bestParam["testAucMean"] = cvRes["testAucMean"]
                self._bestParam["testAucStd"] = cvRes["testAucStd"]
                bestAucMean = cvRes["testAucMean"]
                
    def __cross_validation(self, n_folds=None, shuffle=True, random_state=0, params=None):
        print("XGBoost cross validation:")
        importance = 0
        params['objective'] = 'binary:logistic'
        params["booster"] = 'gbtree'
        params["nthread"] = -1
        params['eval_metric'] = 'auc'
        params["silent"] = self._verbose
        print("-------------------------------------------")
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 5))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("-------------------------------------------")
            print("Start fold {}:".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=20, verbose=self._verbose)
            score = clf.evals_result()
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            y_test_pred = clf.predict_proba(self._dataTest.values)[:, 1]
            
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["auc"][clf.best_ntree_limit-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            reportTmp[fold, 3] = roc_auc_score(self._dataTestLabel.values, y_test_pred)
            reportTmp[fold, 4] = clf.best_ntree_limit-1
            importance += clf.feature_importances_
            print("Test score:{}, valid score:{}".format(reportTmp[fold, 3], reportTmp[fold, 2]))
        report = DataFrame(reportTmp, columns=["fold", "trainAuc", "validAuc", "testAuc", 'bestIteration'])
        return importance/n_folds, report
        print("-------------------------------------------\n")
        
    def refit(self, test):
        bestAuc = self._bestParam["cvReport"]["validAuc"].max()
        bestIterations = int(self._bestParam["cvReport"]["bestIteration"][self._bestParam["cvReport"]["validAuc"] == bestAuc].values)
        params = self._bestParam["param"]
        params['objective'] = 'binary:logistic'
        params["booster"] = 'gbtree'
        params["nthread"] = -1
        params['eval_metric'] = 'auc'
        params["silent"] = self._verbose
        params["n_estimators"] = bestIterations
        clf =  xgb.XGBClassifier(**params)
        clf.fit(self._dataTrain.values, self._dataTrainLabel.values, verbose=False)
        y_pred = clf.predict_proba(test, ntree_limit=bestIterations)[:, 1]
        self.save_refit_model(clf, "..//TrainedModel//xgboost.pkl")
        return y_pred
    
    def save_refit_model(self, data, fileName=None):
        self.__save_model(data, fileName)
        
    def __save_model(self, data, fileName=None):
        assert fileName, "Invalid file name !"
        print("-------------------------------------")
        print("Save file to {}".format(fileName))
        f = open(fileName, 'wb')
        pickle.dump(data, f)
        f.close()
        print("Save successed !")
        print("-------------------------------------")

###############################################################################
class RandomSearchCVLightGBM():
    def __init__(self, dataTrain=None, dataTrainLabel=None, n_estimators=None, stratified=True, paramGrid=None, verbose=False, randomState=2018):
        self._dataTrain, self._dataTest, self._dataTrainLabel, self._dataTestLabel = train_test_split(dataTrain, dataTrainLabel,
                                                                                                      test_size=0.15)
        self._stratified = stratified
        self._verbose = verbose
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._paramGrid = paramGrid
        self._report = {}
        self._estimators = n_estimators
        
    def get_score(self):
        return self._importance, self._report
    
    def random_search_cv(self, n_folds=3, random_state=0):
        print("-------------------------------------------")
        self._bestParam = {}
        bestAucMean = 0
        
        for paramInd in range(self._estimators):
            print("\nNow is {}:".format(paramInd))
            cvRes = {}
            cvRes["param"] = {k: random.sample(list(v), 1)[0] for k, v in self._paramGrid.items()}
            importance, cvReport = self.__cross_validation(n_folds=n_folds, random_state=random_state, params=cvRes["param"])
            cvRes["cvReport"] = cvReport
            cvRes["validAucMean"] = cvReport["validAuc"].mean()
            cvRes["validAucStd"] = cvReport["validAuc"].std()
            cvRes["testAucMean"] = cvReport["testAuc"].mean()
            cvRes["testAucStd"] = cvReport["testAuc"].std()
            self._report[paramInd] = cvRes
            self._importance["param_" + str(paramInd)] = importance
            if cvRes["validAucMean"] > bestAucMean:
                self._bestParam["param"] = cvRes["param"]
                self._bestParam["paramInd"] = paramInd
                self._bestParam["cvReport"] = cvReport
                self._bestParam["validAucMean"] = cvRes["validAucMean"]
                self._bestParam["validAucStd"] = cvRes["validAucStd"]
                self._bestParam["testAucMean"] = cvRes["testAucMean"]
                self._bestParam["testAucStd"] = cvRes["testAucStd"]
                bestAucMean = cvRes["testAucMean"]
                
    def __cross_validation(self, n_folds=None, shuffle=True, random_state=0, params=None):
        print("LightGBM cross validation:")
        importance = 0
        params['objective'] = 'binary'
        params["boosting_type"] = 'gbdt'
        params["nthread"] = -1
        params['subsample_feq'] = 1
        print("-------------------------------------------")
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 5))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("-------------------------------------------")
            print("Start fold {}:".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=200, verbose=self._verbose)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            y_test_pred = clf.predict_proba(self._dataTest.values)[:, 1]
            
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["binary_logloss"][-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            reportTmp[fold, 3] = roc_auc_score(self._dataTestLabel.values, y_test_pred)
            reportTmp[fold, 4] = clf.best_iteration_
            importance += clf.feature_importances_
            print("Test score:{}, valid score:{}".format(reportTmp[fold, 3], reportTmp[fold, 2]))
        report = DataFrame(reportTmp, columns=["fold", "binaryLogloss", "validAuc", "testAuc", 'bestIteration'])
        return importance/n_folds, report
        print("-------------------------------------------\n")
        
    def refit(self, test):
        bestAuc = self._bestParam["cvReport"]["validAuc"].max()
        bestIterations = int(self._bestParam["cvReport"]["bestIteration"][self._bestParam["cvReport"]["validAuc"] == bestAuc].values)
        params = self._bestParam["param"]
        params['objective'] = 'binary'
        params["boosting_type"] = 'gbdt'
        params["nthread"] = -1
        params['subsample_feq'] = 1
        clf = lgb.LGBMClassifier(**params)
        clf.fit(self._dataTrain.values, self._dataTrainLabel.values)
        y_pred = clf.predict_proba(test, num_iteration=bestIterations)[:, 1]
        return y_pred
    
    def save_refit_model(self, data, fileName=None):
        self.__save_model(data, fileName)
        
    def __save_model(self, data, fileName=None):
        assert fileName, "Invalid file name !"
        print("-------------------------------------")
        print("Save LightGBM to {}".format(fileName))
        f = open(fileName, 'wb')
        pickle.dump(data, f)
        f.close()
        print("Save successed !")
        print("-------------------------------------")
###############################################################################
class Stacking():
    def __init__(self, dataTrain=None, dataTrainLabel=None, dataToPredict=None, baseModelPath=None, folds=3, randomState=2018):
        '''
        dataTrain: numpy array
        dataTrainLabels: numpy array
        dataToPredict: Pandas DataFrame, kaggle submission file
        baseModelPath: List, it contains baseModel source path, and the base model is saved in .pkl file
        '''
        self._dataTrain = dataTrain
        self._dataTrainLabel = dataTrainLabel
        self._dataToPredict = dataToPredict
        self._randomState = randomState
        self._baseModelPath = baseModelPath
        self._folds = folds    
    
    def fit_predict(self):
        X = self._dataTrain
        y = self._dataTrainLabel
        X_test = self._dataToPredict
        
        folds = StratifiedKFold(n_splits=self._folds, shuffle=True, random_state=self._randomState)
        self._baseModel = self.load_base_model()
        
        S_train = np.zeros((X.shape[0], len(self._baseModel)))
        S_test = np.zeros((X_test.shape[0], len(self._baseModel)))
        
        for ind, clf in enumerate(self._baseModel):
            print(("\n-------------------------------------"))
            print("Learner {}, path {}:".format(ind, self._baseModelPath[ind]))
            S_test_tmp = np.zeros((X_test.shape[0], self._folds))
            for fold, (trainId, validationId) in enumerate(folds.split(X, y)):
                print("Learner fold {}".format(fold))
                X_train, y_train = X[trainId], y[trainId]
                X_valid, _ = X[validationId], y[validationId]
                
                clf.fit(X_train, y_train)
                S_train[validationId, ind] = clf.predict_proba(X_valid)[:, 1]
                S_test_tmp[:, fold] = clf.predict_proba(X_test)[:, 1]
            S_test[:, ind] = S_test_tmp.mean(1)
        print(("-------------------------------------"))
        lr_results = self.logistic_regression(S_train, y, n_iters=200)
        y_pred = lr_results["bestEstimator"].predict_proba(S_test)[:, 1]
        return y_pred
    
    def logistic_regression(self, X_train, y_train, n_iters=10):
        # Random Search for 2nd level model
        lr = LogisticRegression(fit_intercept=True, max_iter=500, penalty='l2', solver='sag')
        C = np.arange(0.0001, 100, 0.1)
        random_state = [1, 2, 3, 4, 5]
        param = {
                "C":C,
                "random_state":random_state
                }
        
        print("===================Training Logistic Regression===================")
        clf = RandomizedSearchCV(estimator=lr,
                                 param_distributions=param,
                                 n_iter=n_iters,
                                 cv=self._folds,
                                 verbose=1,
                                 n_jobs=-1)
        clf.fit(X_train, y_train)
        print("==================================================================")
        lr_results = {}
        lr_results["bestEstimator"] = clf.best_estimator_
        lr_results["bestCvScore"] = clf.best_score_
        lr_results["bestParam"] = clf.best_params_
        return lr_results
    
    def load_base_model(self):
        return self.__load_base_model()
    
    def __load_base_model(self):
        assert len(self._baseModelPath), "Invalid file path !"
        baseModel = []
        print("-------------------------------------")
        for ind in range(len(self._baseModelPath)):
            pathTmp = self._baseModelPath[ind]
            print("Load file from {}".format(pathTmp))
            f = open(pathTmp, 'rb')
            baseModel.append(pickle.load(f))
            f.close()
        print("Load successed !")
        print("-------------------------------------")
        return baseModel

###############################################################################
def PR_curve(y_real, y_prob, colorCode='b', name=None):
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    plt.step(recall, precision, color='k', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color=colorCode)
    prAuc = auc(recall, precision)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve and auc is {:4f}".format(prAuc))
    
def ROC_curve(y_real, y_prob, colorCode='b'):
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    rocAuc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve and auc = {:4f}".format(rocAuc))

def print_memory_info(data):
	print("============================================")
	print(data.info(memory_usage='deep'))
	print("============================================")

def GBDT(X_train, y_train):
    gbdt = GradientBoostingClassifier()
    n_estimators = [i for i in range(100, 1000)]
    learning_rate = np.arange(0.01, 1, 0.1)
    max_depth = [i for i in range(3, 50)]
    min_samples_split = [i for i in range(2, 30)]
    min_samples_leaf = [i for i in range(1, 30)]
    subsample = np.arange(0.8, 1, 0.02)
    max_features = ['sqrt', 'log2', None]
    random_state = [1, 20, 300, 400, 500]
    
    param = {
            "n_estimators":n_estimators,
            'learning_rate':learning_rate,
            "max_depth":max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'subsample':subsample,
            'max_features':max_features,
            "random_state":random_state
            }
    print("===================Training GBDT Classifier===================")
    clf = RandomizedSearchCV(estimator=gbdt,
                             param_distributions=param,
                             n_iter=1,
                             cv=4,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X_train, y_train)
    print("==================================================================")
    gbdt_results = {}
    gbdt_results["bestEstimator"] = clf.best_estimator_
    gbdt_results["bestCvScore"] = clf.best_score_
    gbdt_results["bestParam"] = clf.best_params_
    
    return gbdt_results

def random_forest(X, y, searchMethod='RandomSearch'): 
    rf = RandomForestClassifier()
    n_estimators = [i for i in range(100, 1000)]
    max_depth = [int(x) for x in range(5, 200, 5)]
    max_features = ('auto', 'sqrt', 'log2', None)
    min_samples_split = [int(x) for x in range(2, 20)]
    min_samples_leaf = [int(x) for x in range(1, 20)]
    param = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            }
    print("------------Training Random Forest--------------")
    clf = RandomizedSearchCV(estimator=rf,
                             param_distributions=param,
                             n_iter=30,
                             cv=5,
                             verbose=1,
                             n_jobs=-1,
                             )
    clf.fit(X, y)
    print("------------------------------------------------")
    rf_results = {}
    rf_results["bestEstimator"] = clf.best_estimator_
    rf_results["bestCvScore"] = clf.best_score_
    rf_results["bestParam"] = clf.best_params_
    return rf_results
