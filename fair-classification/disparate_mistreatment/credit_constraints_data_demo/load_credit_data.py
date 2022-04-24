#load_credit_data.py
from __future__ import division
import urllib.request as urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
import matplotlib.pyplot as plt
# sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    The code will look for the data file in the present directory, if it is not found, it will download them from GitHub.
"""


# In[35]:



def load_credit_data():
    FEATURES_CLASSIFICATION =['AMT_INCOME_TOTAL', 'FLAG_MOBIL', 'FLAG_EMAIL', 
       'MARITAL_STATUS', 'OWN_CAR', 'OWN_PROPERTY', 'CODE_GENDER', 'AGE',
       'EMPLOYED_YEAR', 'EDU_Higher education', 'EDU_Incomplete higher',
       'EDU_Lower secondary', 'EDU_Secondary / secondary special',
       'HOU_House / apartment', 'HOU_Municipal apartment',
       'HOU_Office apartment', 'HOU_Rented apartment', 'HOU_With parents',
       'FAM_1.0', 'FAM_2.0', 'FAM_3.0'] # 21 features to be used for classification
    CONT_VARIABLES = ["AMT_INCOME_TOTAL",'AGE','EMPLOYED_YEAR']
    CLASS_FEATURE = "CREDIT_APPROVAL_STATUS" # the decision variable
    SENSITIVE_ATTRS = ["CODE_GENDER"]


    # load the data 
    application = pd.read_csv("application_record.csv")
    credit_record = pd.read_csv("credit_record.csv")
    convert_to = {'C' : 'Good_Debt', 'X' : 'Neutral_Debt', '0' : 'Neutral_Debt', '1' : 'Neutral_Debt', '2' : 'Neutral_Debt', '3' : 'Bad_Debt', '4' : 'Bad_Debt', '5' : 'Bad_Debt'}
    credit_record.replace({'STATUS' : convert_to}, inplace=True)
    credit_record = credit_record.value_counts(subset=['ID', 'STATUS']).unstack(fill_value=0)
    credit_record['CREDIT_APPROVAL_STATUS'] = -1
    credit_record.loc[(credit_record['Good_Debt'] > 0), 'CREDIT_APPROVAL_STATUS'] = 1
    credit_record['CREDIT_APPROVAL_STATUS'] = credit_record['CREDIT_APPROVAL_STATUS'].astype('int')
    credit_record.drop(['Bad_Debt', 'Good_Debt', 'Neutral_Debt'], axis=1, inplace=True)
    data = application.merge(credit_record, how='inner', on=['ID'])
    data["EMPLOYED_YEAR"]= -data["DAYS_EMPLOYED"]//365
    #Delete abnormal value
    data = data[data["EMPLOYED_YEAR"]>-1]
    # Family members combine >3 
    data.loc[data["CNT_FAM_MEMBERS"]>3,"CNT_FAM_MEMBERS"]=0
    print(data.columns)
    


    """ Formatting the data """
    # Convert days to year
    data['AGE'] = np.ceil(pd.to_timedelta(data['DAYS_BIRTH'], unit='D').dt.days / -365.25)
    data["EMPLOYED_YEAR"]= -data["DAYS_EMPLOYED"]//365
    data["CODE_GENDER"] = np.where(data["CODE_GENDER"] == 'M',1,-1)
    
    #Delete abnormal value
    data = data[data["EMPLOYED_YEAR"]>-1]
    # dummy variables
    edu_dummy = pd.get_dummies(data['NAME_EDUCATION_TYPE'], prefix='EDU',drop_first= True)
    edu_dummy.replace(0,"-1",inplace=True)
    House_dummy = pd.get_dummies(data['NAME_HOUSING_TYPE'], prefix='HOU',drop_first= True)
    House_dummy.replace(0,"-1",inplace=True)
    famnum_dummy = pd.get_dummies(data['CNT_FAM_MEMBERS'], prefix='FAM',drop_first= True)
    famnum_dummy.replace(0,"-1",inplace=True)
    data = pd.concat([data,edu_dummy.astype(int),House_dummy.astype(int), famnum_dummy.astype(int)], axis=1)
    
    # Binary encoding
    # Past over due is 1
    # Married is 1
    data["MARITAL_STATUS"] = np.where((data["NAME_FAMILY_STATUS"]=='Civil marriage') | (data["NAME_FAMILY_STATUS"]=='Married'),1,-1 )
    # Having a car is 1
    data["OWN_CAR"] = np.where(data["FLAG_OWN_CAR"]=="Y",1,-1)
    # Having a property is 1
    data["OWN_PROPERTY"] = np.where(data["FLAG_OWN_REALTY"]=="Y",1,-1)

     

    # Convert 0 to -1
    data.loc[data["FLAG_MOBIL"]==0,"FLAG_MOBIL"]=-1
    data.loc[data["FLAG_EMAIL"]==0,"FLAG_EMAIL"]=-1
    
    #Clear unnecessary columns
    data = data[FEATURES_CLASSIFICATION+[CLASS_FEATURE]]
    
    # convert to np array
    data = data.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    y = data[CLASS_FEATURE]  
    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    """ Feature normalization"""
    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance  
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col
        
        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals


        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

#     """permute the date randomly"""
#     perm = list(range(0,X.shape[0]))
#     shuffle(perm)
#     X = X[perm]
#     y = y[perm]
#     for k in x_control.keys():
#         x_control[k] = x_control[k][perm]


    X = ut.add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert(len(feature_names) == X.shape[1])
    print("Features we will be using for classification are:", feature_names, "\n")


    return X, y, x_control

