import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
plt.style.use('fivethirtyeight')
from scipy import stats
%matplotlib inline

train = pd.read_csv('../MyPC/Ames/train.csv')
test = pd.read_csv('../MyPC/Ames/test.csv')

missing = train.isna().sum()
missing = missing[missing>0]
missing_perc = missing/train.shape[0]*100
na = pd.DataFrame([missing, missing_perc], index = ['missing_num', 'missing_perc']).T
na = na.sort_values(by = 'missing_perc', ascending = False)

sns.pairplot(x_vars = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'], y_vars = 'SalePrice', data = train)
plt.show()
sns.pairplot(x_vars = [ 'FireplaceQu','GarageType', 'GarageYrBlt', 'GarageFinish'],  y_vars = 'SalePrice', data = train)
plt.show()
sns.pairplot(x_vars = ['BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType'], y_vars = 'SalePrice', data = train)
plt.show()
plt.savefig('myfig.jpg')

#Dealing with the missing values
train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageYrBlt', 
            'GarageFinish','BsmtExposure', 'BsmtFinType2', 'BsmtFinType1',
            'MasVnrType'], axis = 1, inplace = True)

train.drop(train[train.Electrical.isna()].index, axis = 0, inplace = True)
NA = [ 'GarageQual', 'GarageCond', 'BsmtCond', 'BsmtQual']
for na in NA:
    train[na].fillna('NA', inplace = True)
train['MasVnrArea'].fillna(0, inplace = True)
train['LotFrontage'].fillna(0, inplace = True)

#Understanding the Corrrelations
sns.distplot(train.SalePrice)
plt.show()

figure = plt.figure(figsize = (13,5))
plt.subplot(1,2,1)
stats.probplot(train.SalePrice, plot = plt)
plt.title('Actual SalePrice')
plt.subplot(1,2,2)
train.SalePrice = np.log(train.SalePrice)
stats.probplot(train.SalePrice, plot = plt)
plt.title('SalePrice after log transformation')
plt.show()
plt.savefig('SalePrice.jpg')
plt.figure(figsize = (15,10))
sns.heatmap(train.corr(), vmax=.8, square=True)
plt.savefig('C:\\Users\\MyPC\\Pictures\\Heatmap.jpg')

plt.figure(figsize = (11,4))
plt.subplot(1,2,1)
plt.scatter(train.OverallQual, train.SalePrice)
plt.xlabel('OverallQual'); plt.ylabel('Sale Price')
plt.subplot(1,2,2)
plt.scatter(train.GrLivArea, train.SalePrice)
plt.xlabel('Grnd Living Area'); plt.ylabel('Sale Price')
plt.show()

var = ['GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
sns.pairplot(x_vars = var, y_vars = 'SalePrice', data = train)
plt.show()

#Feature Engineering
train.drop('Id', axis = 1, inplace = True)
train.drop(['TotRmsAbvGrd', 'GarageArea'], axis =1, inplace = True)
features = ['OverallQual', 'GrLivArea']
for feat in features:
    train[feat+'_p2'] = train[feat] **2
    train[feat+'_p3'] = train[feat] **3
mp = {'Ex': 5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}
for feat in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual','GarageQual', 'GarageCond', ]:
    train[feat] = train[feat].map(mp) 
mp = {'N':0, 'Y':2 , 'P':1}
for feat in ['CentralAir', 'PavedDrive']:
    train[feat] = train[feat].map(mp)
mp = {'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1}
train['Functional'] = train['Functional'].map(mp)
mp = {'Gtl':1 ,'Mod':2 , 'Sev':3}

train['LandSlope'] = train['LandSlope'].map(mp)  
train['TotBath'] = train['BsmtFullBath']+train['FullBath']+.5*(train.BsmtHalfBath+train.HalfBath)
train['Overall_Score'] = train.OverallQual*train.OverallCond
train['Total_area'] = train['1stFlrSF']+train['2ndFlrSF']+train.TotalBsmtSF
train['Garage_Score'] = train.GarageQual*train.GarageCond
train['Kitchen_Score'] = train.KitchenAbvGr*train.KitchenQual
train['Bsmt_Score'] = train.BsmtQual*train.BsmtCond

train = pd.get_dummies(train)

train_x = train.drop('SalePrice', axis = 1)
train_y = pd.DataFrame(train.SalePrice)
index = train_x.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
train_x = pd.DataFrame(train_x, columns = index)

#Doing the same for the test data
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish','BsmtExposure', 'BsmtFinType2', 'BsmtFinType1','MasVnrType'], axis = True, inplace = True)
test.drop(test[test.Electrical.isna()].index, axis = 0, inplace = True)
NA = [ 'GarageQual', 'GarageCond', 'BsmtCond', 'BsmtQual']
for na in NA:
    test[na].fillna('NA', inplace = True)
    NA = ['GarageQual', 'GarageCond','BsmtCond', 'BsmtQual']
for na in NA:
    test[na].fillna('NA', inplace = True)
fill_zero = ['GarageCars', 'GarageArea']
for zeros in fill_zero:
    test[zeros].fillna(0, inplace = True)
test['MasVnrArea'].fillna(0, inplace = True)
test['LotFrontage'].fillna(0, inplace = True)
test['Electrical'].fillna('SBrkr', inplace = True)
test.MSZoning.fillna('RL', inplace = True)
test.Utilities.fillna('AllPub', inplace = True)
test.BsmtFinSF1.fillna(0.0, inplace = True)
test.BsmtFinSF2.fillna(0.0, inplace = True)
test.BsmtUnfSF.fillna(0.0, inplace = True)
test.TotalBsmtSF.fillna(0.0, inplace = True)
test.BsmtFullBath.fillna(0.0, inplace = True)
test.BsmtHalfBath.fillna(0.0, inplace = True)
test.Functional.fillna('Typ', inplace = True)
test.Exterior1st.fillna('VinylSd', inplace = True)
test.Exterior2nd.fillna('VinylSd', inplace = True)
test.KitchenQual.fillna('TA', inplace = True)
test.SaleType.fillna('WD', inplace = True)
test.drop('Id', axis = 1, inplace = True)
features = ['OverallQual', 'GrLivArea']
for feat in features:
    test[feat+'_p2'] = test[feat] **2
    test[feat+'_p3'] = test[feat] **3
test.drop(['TotRmsAbvGrd', 'GarageArea'], axis =1, inplace = True)
mp = {'Ex': 5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}
for feat in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']:
    test[feat] = test[feat].map(mp) 
mp = {'N':0, 'Y':2 , 'P':1}
for feat in ['CentralAir', 'PavedDrive']:
    test[feat] = test[feat].map(mp)
mp = {'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1}
test['Functional'] = test['Functional'].map(mp)
mp = {'Gtl':1 ,'Mod':2 , 'Sev':3}

test['LandSlope'] = test['LandSlope'].map(mp)
test['TotBath'] = test['BsmtFullBath'] + test['FullBath'] + 0.5*(test.BsmtHalfBath + test.HalfBath)
test['Overall_Score'] = test.OverallQual*test.OverallCond
test['Total_area'] = test['1stFlrSF']+test['2ndFlrSF']+test.TotalBsmtSF
test['Garage_Score'] = test.GarageQual*test.GarageCond
test['Kitchen_Score'] = test.KitchenAbvGr*test.KitchenQual
test['Bsmt_Score'] = test.BsmtQual*test.BsmtCond
test = pd.get_dummies(test)

testcol = test.columns.tolist()
traincol = train_x.columns.tolist()
diff = list(set(traincol).difference(testcol))
last_cols = train_x[diff]
train_x.drop(diff, axis =1, inplace = True)
index = test.columns
scaler = StandardScaler()
test = scaler.fit_transform(test)
test = pd.DataFrame(test, columns = index)

#Model Creation
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mse
model1 = Ridge(alpha = 1)
model2 = LinearRegression()
model2.fit(train_x, train_y)
model1.fit(train_x, train_y)
print("THe accuracy of training LR set is ", model2.score(train_x, train_y))
predRigde= model1.predict(test)
predLR= model2.predict(test)
print("THe accuracy of training Ridge set is ", model1.score(train_x, train_y))
