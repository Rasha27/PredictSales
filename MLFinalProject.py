!pip install -U -q PyDrive
import os

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from oauth2client.client import GoogleCredentials
from google.colab import auth

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import pyplot
%matplotlib inline

import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Authenticat PyDrive client.
auth.authenticate_user()
google_auth = GoogleAuth()
google_auth.credentials = GoogleCredentials.get_application_default()
# Creating PyDrive Client
google_drive = GoogleDrive(google_auth)

# Listing all the files in the current directory of drive
files_list = google_drive.ListFile(
    {'q': "'1ah6TPxFUi3K6U1pvHy74uJVT0EN3ywNj' in parents"}).GetList()
for file in files_list:
  print('title: %s, id: %s' % (file['title'], file['id']))

#loading data from drive into goole collab local storage
downloadPath = os.path.expanduser('~/data')
try:
  os.makedirs(downloadPath)
except FileExistsError:
  pass

itemsFile = os.path.join(downloadPath, 'items-translated.csv')
items = google_drive.CreateFile({'id': '1DdJ-EGjo2QBDysR8zD7MayYgFNwgtwtd'})
items.GetContentFile(itemsFile)

itemsCategoriesFile = os.path.join(downloadPath, 'item_categories-translated.csv')
itemCategories = google_drive.CreateFile({'id': '1jKrFPDkHJp-y-i_SwgVDfarlAqYuV1Ud'})
itemCategories.GetContentFile(itemsCategoriesFile)

shopsFile = os.path.join(downloadPath, 'shops-translated.csv')
shops = google_drive.CreateFile({'id': '10g2dZpBPdeyxmP0BzJpi-L5PEutn0zTE'})
shops.GetContentFile(shopsFile)

trainFile = os.path.join(downloadPath, 'sales_train.csv')
train = google_drive.CreateFile({'id': '1NYUCyRFfBVXSBp3GQaqQa5z51i_nhnAL'})
train.GetContentFile(trainFile)

testFile = os.path.join(downloadPath, 'test.csv')
test = google_drive.CreateFile({'id': '15_cHTMnvB1uUnu-5IC8i1dOBhmStCWUU'})
test.GetContentFile(testFile)

itemsData = pd.read_csv(itemsFile)
itemsData.head()

itemsCategoriesData = pd.read_csv(itemsCategoriesFile)
itemsCategoriesData.head()

shopsData = pd.read_csv(shopsFile)
shopsData.head()

trainData = pd.read_csv(trainFile)
trainData.head()

fig,ax = plt.subplots(figsize=(15,7))
a = trainData.groupby(['shop_id']).sum()['item_cnt_day'].plot(ax=ax)

testData = pd.read_csv(testFile)
testData.head()

#function to check if test data contains in train or not and removing duplicate entries in training data
def preprocessData(train,test):
  
  #replacing NA with 0
  train['item_price'].fillna(0, inplace=True)
  #formatting date
  train['date'] = pd.to_datetime(train.date, format="%d.%m.%Y")
  
  
  test_shopid = test.shop_id
  unique_shopid = np.unique(test_shopid)
  train = train[train.shop_id.isin(unique_shopid)]
  test_itemId = test.item_id
  unique_itemId = np.unique(test_itemId)
  train = train[train.item_id.isin(unique_itemId)]
  
  set = {'date','shop_id','item_id'}
  train = train.drop_duplicates(set,keep ='first')
  
  # remove any outliers present in train data, for that we have used mean and SD
  plt.figure(figsize=(12,6))
  sb.boxplot(x=train.item_cnt_day)
  
  item_cnt_mean, item_cnt_std = mean(train.item_cnt_day), std(train.item_cnt_day)
  cut_off = item_cnt_std * 3
  lowerBound, upperBound = item_cnt_mean - cut_off, item_cnt_mean + cut_off
  
  train = train[(train.item_cnt_day < upperBound) & (train.item_cnt_day > lowerBound)]
  
  #clustering categories
  dict = {'accessories' : 1,'game consoles' : 3,'games' : 4, 'programs' : 5,'gifts' : 6,'music' : 7,'books' :8 ,'cinema' :9,'payment cards' : 10,'pc games' : 11}
  categories  = itemsCategoriesData["item_category_name_translated"];
  cluster = []
  count = 12
  for i in categories:
    word = i.split("-")[0].strip().lower()
    value = dict.get(word)
    if value==None:
      cluster.append(count)
      count+=1
    else:
      cluster.append(value)
  itemsCategoriesData["categories_grouped"] = cluster
  
  #adding category data with train data
  train  = pd.pivot_table(train, index=['shop_id', 'item_id'],values='item_cnt_day', columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
  itemCats = itemsData.merge(itemsCategoriesData, left_on='item_category_id', right_on='item_category_id',how = 'inner')[['item_id','categories_grouped']]
  
  result = train.merge(itemCats,left_on='item_id',right_on='item_id',how ='inner')
  
  train = result[['categories_grouped'] + ['shop_id', 'item_id'] + list(range(34))]
  
  #preprocessing testdata 
  testdata = testData[['shop_id', 'item_id']]
  test_mergeData = testdata.merge(train, on = ["shop_id", "item_id"], how = "left").fillna(0)
  #print(train)
  return train,test_mergeData
processedtrainDf,processedtestDf = preprocessData(trainData, testData)

def linearRegression(xTrain,yTrain,xTest):
  """
  params_cons = {
      'fit_intercept' : [True, False],
      'normalize' : [True, False],
      'copy_X' : [True, False],
      'n_jobs' : [2,3,4],
  }
  """
  param={
      'normalize': [True],
      'n_jobs': [2], 
      'fit_intercept': [False],
      'copy_X': [True]
  }
  linear_reg = RandomizedSearchCV(LinearRegression(),param_distributions =param )
  linear_reg.fit(xTrain, yTrain)
  #print(pre_gs_inst.best_params_)
  
  yTrainPredicted = linear_reg.predict(xTrain)
  yTestPredcted = linear_reg.predict(xTest)
  print(pd.DataFrame(yTestPredcted).describe())
  
  return yTrainPredicted
def randomForest(xTrain, yTrain, xTest):
  rfReg = RandomForestRegressor()
  
  """
  Parameters used for initial tuning
  randomParams = {'n_estimators': [20, 50, 60],
                'max_features' : ['auto', 'sqrt', 'log2'],
                'min_samples_split' : [2, 4, 8],
                'min_weight_fraction_leaf' :[0.1, 0.2, 0.5],
                'max_features' : [2, 6, 8],
                'min_samples_leaf': [1, 3, 4],]
                'bootstrap' : ['True', 'False']
              }
  """
  randomParams = {'n_estimators': [20], 
                'max_features' : ['sqrt'],
                'min_samples_split': [8],
                'min_weight_fraction_leaf': [0.1], 
                'max_features': [8], 
                'min_samples_leaf': [3],
                'bootstrap': ['True']
               }
  
  rfRandom = RandomizedSearchCV(estimator = rfReg, param_distributions = randomParams, n_iter = 500, cv = 3, verbose=2, random_state=20, n_jobs = -1)
  rfRandom.fit(xTrain,yTrain)
  #print(rfRandom.best_params_)
  
  #yTrain are the values to be predicted
  yTrainPredicted = rfRandom.predict(xTrain)
  yTestPredcted = rfRandom.predict(xTest)
  print(pd.DataFrame(yTestPredcted).describe())
  
  return yTrainPredicted

def xgboost(xTrain,yTrain,xTest):
  """
  parameters = {'objective':['reg:linear'],
              'nthread':[4],
              'max_depth': [5, 6, 7, 8],
              'learning_rate': [.03, 0.05, .07],
              'silent': [1],
              'min_child_weight': [4],
              'colsample_bytree': [0.7],
              'subsample': [0.7],
              'n_estimators': [500]}
  """
  parameters = {'objective':['reg:linear'],
              'nthread':[4],
              'max_depth': [8],
              'learning_rate': [.03],
              'silent': [1],
              'min_child_weight': [4],
              'colsample_bytree': [0.7],
              'subsample': [0.7],
              'n_estimators': [500]}
  
  xgbRegressor = xgb.XGBRegressor()
  xgb_grid = RandomizedSearchCV(xgbRegressor, parameters, cv = 5, n_jobs = -1, verbose=True)
  xgb_grid.fit(xTrain,yTrain)
  #print(xgb_grid.best_params_)
  
  #yTrain are the values to be predicted
  yTrainPredicted = xgb_grid.predict(xTrain)
  yTestPredcted = xgb_grid.predict(xTest)
  print(pd.DataFrame(yTestPredcted).describe())
  
  return yTrainPredicted

#tuning parameters with decision Tree algorithm

def decisionTree(xTrain,yTrain,xTest):
  
  regression_model = DecisionTreeRegressor(criterion="mse",splitter = 'best',max_depth = 12 ,min_samples_leaf=5,max_features = 'log2')
  
  regression_model.fit(xTrain,yTrain)
  predictions = regression_model.predict(xTrain)
  test_preds = regression_model.predict(xTest)
  print(test_preds)
  print(pd.DataFrame(test_preds).describe())
  
  return predictions

def adaBoost(xTrain,yTrain,xTest):
  """
  params_cons = {
      'n_estimators': [20, 30,50],
      'learning_rate' : [0.01,0.1,0.5],
      'loss' : ['linear', 'square', 'exponential']
  }
  """
  
  actual_param = {
      'learning_rate' : [0.01],
      'n_estimators': [50] ,
      'loss' : ['exponential'],  
      }
  adRandom = RandomizedSearchCV(AdaBoostRegressor(),actual_param,scoring='neg_mean_squared_error', n_jobs = -1, cv=5)
  adRandom.fit(xTrain, yTrain)
  
  predictions = adRandom.predict(xTrain)
  test_preds = adRandom.predict(xTest)
  
  print(test_preds)
  print(pd.DataFrame(test_preds).describe())
  
  return predictions

#calculating errorMetrics for different models
def errorMetrics(yTrain,predictions):
  # mean absolute error
  print(" Mean absolute error      :" , mean_absolute_error(yTrain, predictions) )
  
  print(" explained varaince score :" , explained_variance_score(yTrain, predictions, multioutput='variance_weighted'))
  
  print(" mean squared error       :" , mean_squared_error(yTrain, predictions, multioutput='uniform_average'))
  
  print(" median absolute error    :" , median_absolute_error(yTrain, predictions))
  
  print(" R2 score                 :" , r2_score(yTrain, predictions, multioutput='uniform_average'))
  
  print(" Root Mean Squared Error  :" , np.sqrt(mean_squared_error(yTrain, predictions, multioutput='uniform_average')))

#filtering columns for train data and test data
xTrain = processedtrainDf.iloc[:,(processedtrainDf.columns != 33)].values
yTrain = processedtrainDf.iloc[:, processedtrainDf.columns == 33].values
xTest =  processedtestDf.iloc[:, processedtestDf.columns != 0].values

print("\n ---------Decision Tree ---------------------")
preds_DT = decisionTree(xTrain,yTrain,xTest)
errorMetrics(yTrain,preds_DT)

print("\n ---------Random Forest ---------------------")
preds_RF = randomForest(xTrain,yTrain,xTest)
errorMetrics(yTrain,preds_RF)

print("\n ---------adaBoost ---------------------")
preds_AD = adaBoost(xTrain,yTrain,xTest)
errorMetrics(yTrain,preds_AD)

print("\n ---------linearregression ---------------------")
preds_LR = linearRegression(xTrain,yTrain,xTest)
errorMetrics(yTrain,preds_LR)

print("\n ---------xgboost ---------------------")
preds_XG = xgboost(xTrain,yTrain,xTest)
errorMetrics(yTrain,preds_XG)
