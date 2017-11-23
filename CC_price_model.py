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

def create_dummies(data_in):
    cols2edit = [i for i in data_in.columns if type(data_in[i].iloc[1]) == str or type(data_in[i].iloc[1]) == float]
    for col_id in cols2edit:
        data_in[col_id].fillna('NULL', inplace = True) #if any nas are present, which there dont appear to be
        data_in = pd.concat((data_in,pd.get_dummies(data_in[col_id], prefix = col_id)), axis = 1) #create dummy variables
        del data_in[col_id]
    return data_in
    
def transf_numeric(data_in):
    #print(data_in['cbd'])
    numeric_feats = data_in.dtypes[data_in.dtypes != "object"].index
    #print(data_in.dtypes)
    #Remove 4 comps as produce error
    rm_list = ['comp_1','comp_2','comp_3','comp_4']
    numeric_feats_conv = [i for i in numeric_feats if i not in rm_list]
    data_in[numeric_feats_conv] = np.log1p(data_in[numeric_feats_conv]) #using log1p on PCA data produces nan. Can convert to num with nan_to_num
    #print(data_in[numeric_feats])
    #scaler = StandardScaler().fit(data_in[numeric_feats])
    #data_in[numeric_feats] = scaler.transform(data_in[numeric_feats])
    #print(data_in[numeric_feats])
    return data_in

def whlsle_prc(insamp_info):
    #load models
    filename = 'LASSO_model.sav'
    lasso = pickle.load(open(filename, 'rb'))
    filename = 'xgb_model.sav'
    regr = pickle.load(open(filename, 'rb'))
    #Preprocess input
    insamp_info['month']=str(insamp_info['month']) #change month to string
    insamp_info=pd.Series(insamp_info)
    insamp_info=insamp_info.to_frame()
    #insamp_info=insamp_info[0]
    #drop unused columns
    colstorem=['date','retailer','retailer_city','processor','processor_city','producer','producer_city','strain_display_name','product_name']
    insamp_info=insamp_info.drop(colstorem)
    #insamp_info=create_dummies(insamp_info)
    #print(insamp_info)
    #insamp_info=transf_numeric(insamp_info)
    #lasso_pred = lasso.predict(insamp_info)
    pplb=1825
    return pplb


