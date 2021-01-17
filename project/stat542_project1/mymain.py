
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import pandas as pd


# In[2]:


train = pd.read_csv('train.csv', index_col = 'PID')
test = pd.read_csv('test.csv', index_col = 'PID')
y_train = np.log(train['Sale_Price'])
data = pd.concat([train.drop('Sale_Price', axis = 1), test])

numeric_col = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add','Mas_Vnr_Area', 
           'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF','Bsmt_Half_Bath','Total_Bsmt_SF', 
           'First_Flr_SF', 'Second_Flr_SF', 'Gr_Liv_Area','Bsmt_Full_Bath', 'Full_Bath',
           'Half_Bath','Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd', 'Fireplaces',
           'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF',
           'Enclosed_Porch', 'Screen_Porch', 'Mo_Sold', 'Year_Sold']

nominal_col = ['MS_SubClass', 'MS_Zoning', 'Alley', 'Lot_Shape', 'Land_Contour', 'Lot_Config', 'Neighborhood', 
           'Condition_1', 'Bldg_Type', 'House_Style', 'Roof_Style', 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type', 
           'Foundation', 'BsmtFin_Type_1', 'BsmtFin_Type_2', 'Central_Air', 'Electrical', 'Functional', 
           'Garage_Type', 'Garage_Finish', 'Paved_Drive', 'Fence', 'Sale_Type', 'Sale_Condition']

ordinal_col = ['Overall_Qual', 'Overall_Cond', 'Exter_Qual', 'Exter_Cond', 'Bsmt_Qual', 'Bsmt_Cond', 
           'Heating_QC', 'Kitchen_Qual', 'Fireplace_Qu', 'Garage_Qual', 'Garage_Cond', 'Bsmt_Exposure']

#winsorization
def winsorize(data):
    data = data.apply(lambda x: x.clip(upper = x.quantile(0.95)))
    return data

#correlation to repsonse variable
def corr(data, y, threshold = 0.1):
    result = [x for x in data.columns if abs(data.join(y).corr()['Sale_Price'][x]) > threshold]
    return result

# merge infrequent levels as other
def merge(data, threshold = 0.01):
    result = data.copy()
    for col in data.columns:
        infrequent = [val for val in result[col].values if result[col].value_counts()[val] < threshold * len(result[col])]
        result[col] = result[col].apply(lambda x: 'Other' if x in infrequent else x)
    result = pd.get_dummies(result)
    return result

# relabel ordinal
def relabel(data):
    relabel_dict = {'Excellent': 4, 'Very_Excellent': 4, 'Very_Good': 4, 
                    'Good': 3, 'Gd': 3,'Above_Average': 3, 
                    'Average': 2, 'Below_Average':2, 'Av': 2, 'Typical': 2,
                    'Fair': 1, 'Poor': 1, 'Very_Poor': 1, 'No_Basement': 1, 
                    'No_Fireplace': 1, 'Mn': 1, 'No': 1, 'No_Garage': 1}
    result = data.copy()
    for col in data.columns:
        result[col] = result[col].apply(lambda x: relabel_dict[x])
    return result

#preprocess
X = merge(data[nominal_col]).join(relabel(data[ordinal_col])).join(winsorize(data[numeric_col])) 

#remove low correlation variables and split train test
X = X[corr(X[~ X.index.isin(test.index)], y_train)]
X_train = X[~ X.index.isin(test.index)]
X_test = X.loc[test.index]

#xgb prediction model
xgb1 = xgb.XGBRegressor(objective ='reg:linear', subsample = 0.7, colsample_bytree = 0.7, learning_rate = 0.1,
                       max_depth = 6, alpha = 10, n_estimators = 100, min_child_weight = 5)
xgb1.fit(X_train, y_train)
y_pred_xgb1 = xgb1.predict(X_test)

xgb2 = xgb.XGBRegressor(objective ='reg:linear', subsample = 0.8, colsample_bytree = 0.8, learning_rate = 0.1,
                       max_depth = 7, alpha = 10, n_estimators = 140, min_child_weight = 6)
xgb2.fit(X_train, y_train)
y_pred_xgb2 = xgb2.predict(X_test)

xgb_result1 = pd.DataFrame(np.around(np.exp(y_pred_xgb1), 1), columns = ['Sale_Price'], index = test.index)
xgb_result2 = pd.DataFrame(np.around(np.exp(y_pred_xgb2), 1), columns = ['Sale_Price'], index = test.index)
xgb_result1.to_csv('mysubmission1.txt')
xgb_result2.to_csv('mysubmission2.txt')

