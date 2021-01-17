
# coding: utf-8

# In[1]:


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.metrics import log_loss


# In[2]:


train = pd.read_csv('train.csv', index_col = 'id')
test = pd.read_csv('test.csv', index_col = 'id')
y_train = (train['loan_status'] == 'Charged Off').apply(np.uint8)
X = pd.concat([train.drop('loan_status', axis = 1), test])


# In[3]:


X['fico_score'] = 0.5*X['fico_range_low'] + 0.5*X['fico_range_high']
X['log_annual_inc'] = np.log(X['annual_inc'] + 1)
X['earliest_cr_line'] = X['earliest_cr_line'].apply(lambda x: int(x[-4:]))
X['emp_length'].replace('10+ years', '10 years', inplace=True)
X['emp_length'].replace('< 1 year', '0 years', inplace=True)
X['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
X['emp_length'] = X['emp_length'].apply(emp_length_to_int)
    
#columns dropped: id, grade, emp_title, title, zip_code, open_acc, revol_bal, total_acc
nominal = ['sub_grade', 'verification_status', 'home_ownership', 'purpose',
           'addr_state', 'initial_list_status', 'application_type','term']
numeric = ['loan_amnt', 'int_rate', 'installment', 'dti', 'earliest_cr_line', 'revol_util',
           'emp_length', 'mort_acc', 'pub_rec_bankruptcies', 'pub_rec','fico_score','log_annual_inc']


# In[4]:


# get_dummies for nominal variables
def preprocess_nominal(data):
    return pd.get_dummies(data)

#Missing value impute and winsorize
def preprocess_numeric(data):
    result = data.copy()
    imputer = Imputer()
    missing_col = result.columns[result.isnull().any()]
    result[missing_col] = imputer.fit_transform(result[missing_col].values)
    result = result.apply(lambda x: x.clip(upper = x.quantile(0.95), lower = x.quantile(0.05)))
    return result

# correlation to repsonse variable
def corr(data, y, threshold = 0.01):
    corr = data.join(y).corr()['loan_status']
    result = [x for x in data.columns if abs(corr[x]) > threshold]
    return result


# In[5]:


#preprocess
X = preprocess_nominal(X[nominal]).join(preprocess_numeric(X[numeric]))


# In[6]:


#xgb prediction model
X = X[corr(X[~ X.index.isin(test.index)], y_train)]
X_train = X[~ X.index.isin(test.index)]
X_test = X.loc[test.index]
xgb_model = xgb.XGBRegressor(objective ='reg:linear', subsample = 0.8, colsample_bytree = 0.8, learning_rate = 0.1,
                       max_depth = 7, n_estimators = 100, min_child_weight = 8)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb[y_pred_xgb < 0] = np.exp(-4)
xgb_result = pd.DataFrame(y_pred_xgb, columns = ['prob'], index = test.index)
xgb_result.to_csv('mysubmission1.txt')

