# Import libraries
import os
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

#load training data
data_in = pd.read_csv('dataset.csv')
data_in = data_in.iloc[:, 2:] #Remove first column as has no info, and second column date as have numeric timestamp

#change month to string
data_in['month']=data_in['month'].astype(str)

#feature engineering
def create_dummies(data_in):
    cols2edit = [i for i in data_in.columns if type(data_in[i].iloc[1]) == str or type(data_in[i].iloc[1]) == float]
    for col_id in cols2edit:
        data_in[col_id].fillna('NULL', inplace = True) #if any nas are present, which there dont appear to be
        data_in = pd.concat((data_in,pd.get_dummies(data_in[col_id], prefix = col_id)), axis = 1) #create dummy variables
        del data_in[col_id]
    return data_in
    
def transf_numeric(data_in):
    numeric_feats = data_in.dtypes[data_in.dtypes != "object"].index
    #Remove 4 comps as produce error
    rm_list = ['comp_1','comp_2','comp_3','comp_4']
    numeric_feats_conv = [i for i in numeric_feats if i not in rm_list]
    data_in[numeric_feats_conv] = np.log1p(data_in[numeric_feats_conv]) #using log1p on PCA data produces nan. Can convert to num with nan_to_num
    rm_list = ['pplb_pretax']
    numeric_feats= [i for i in numeric_feats if i not in rm_list]
    scaler = StandardScaler().fit(data_in[numeric_feats])
    data_in[numeric_feats] = scaler.transform(data_in[numeric_feats])
    return data_in
    
#Preprocess data
data_in=create_dummies(data_in)
data_in=transf_numeric(data_in)
pplb_train=data_in.pplb_pretax
data_train=data_in.drop('pplb_pretax', axis=1)
del data_in

#LASSO regression
alpha_ridge = [1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
coeffs = {}
for alpha in alpha_ridge:
    r = Lasso(alpha=alpha, normalize=True, max_iter=100000000)
    r = r.fit(data_train, pplb_train)
    
grid_search = GridSearchCV(Lasso(alpha=alpha, normalize=True), scoring='neg_mean_squared_error',
                           param_grid={'alpha': alpha_ridge}, cv=10, n_jobs=-1)
grid_search.fit(data_train, pplb_train)
lasso = Lasso(alpha=.00001, normalize=True, max_iter=1e6)
lasso = lasso.fit(data_train, pplb_train)
# save model
filename = 'LASSO_model.sav'
pickle.dump(lasso, open(filename, 'wb'))

#XGBoost Model
regr = xgb.XGBRegressor(
                 colsample_bytree=0.3,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=1668,                                                                  
                 reg_alpha=1,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
                 
regr.fit(data_train, pplb_train)
# save model
filename = 'xgb_model.sav'
pickle.dump(regr, open(filename, 'wb'))

