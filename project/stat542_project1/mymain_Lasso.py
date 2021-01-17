
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# In[2]:


train = pd.read_csv('train.csv', index_col = 'PID')
test = pd.read_csv('test.csv', index_col = 'PID')
data = pd.concat([train.drop('Sale_Price', axis = 1), test])
y_train = np.log(train['Sale_Price'])

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

def one_step_lasso(r, x, lam):
    xx = np.dot(x, x)
    xr = np.dot(r, x)
    b = (abs(xr) - lam/2) / xx
    b = np.sign(xr) * ((b > 0) * b)
    
    return b

def my_lasso(X, y, lam, n_iter = 200, standardize  = True):
    p = len(X[0])
    if standardize:
        ss = StandardScaler()
        X = ss.fit_transform(X)
        x_mean = ss.mean_
        x_scale = ss.var_
        
        y = ss.fit_transform(y)
        y_mean = ss.mean_
        y_scale = ss.var_
    
    b = np.zeros((p, 1))
    r = y.reshape(-1)

    for step in range(n_iter):
        for j in range(p):
            r = r + X[:, j] * b[j]
            b[j] = one_step_lasso(r, X[:, j], lam)
            r = r - X[:, j] * b[j]
    
    b1 = b.reshape(-1) * np.sqrt(y_scale) / np.sqrt(x_scale)
    b0 = y_mean[0] - np.dot(x_mean, b1)
    
    return [b0, b1]



# In[3]:


#preprocess
X = merge(data[nominal_col]).join(relabel(data[ordinal_col])).join(winsorize(data[numeric_col])) 

#remove low correlation variables and split train test
X = X[corr(X[~ X.index.isin(test.index)], y_train)]
X_train = X[~ X.index.isin(test.index)].values
y_train_lasso = y_train.values.reshape(-1, 1)
X_test = X.loc[test.index].values

# lambda = 14 is optimal 
[b0, b1] = my_lasso(X_train, y_train_lasso, 14)
lasso_y_pred = np.dot(X_test, b1) + b0
lasso_result = pd.DataFrame(np.around(np.exp(lasso_y_pred), 1), columns = ['Sale_Price'], index = test.index)
lasso_result.to_csv('mysubmission3.txt')

