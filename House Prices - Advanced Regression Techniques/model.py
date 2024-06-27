import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

dataset = pd.read_csv('train.csv')
y = dataset['SalePrice']
X = dataset.drop(columns=['Id', 'SalePrice'])

categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
                        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
                        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
                        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
                        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                        'SaleType', 'SaleCondition']

numerical_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                      '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
                      'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
                      'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
                      'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 
                      'MoSold', 'YrSold']

imputer = SimpleImputer(strategy='mean')
imputer.fit(X[numerical_features])
X[numerical_features] = imputer.transform(X[numerical_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
cat_imputer.fit(X[categorical_features])
X[categorical_features] = cat_imputer.transform(X[categorical_features])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
X = ct.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

regressor = XGBRegressor(n_jobs=4, random_state=1)

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
random_search.fit(X_train, y_train)

best_regressor = random_search.best_estimator_
y_pred = best_regressor.predict(X_test)

mse = mean_squared_error(np.log(y_test), np.log(y_pred))
rmse = np.sqrt(mse)
print("RMSE:", rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2:", r2)

test_dataset = pd.read_csv('test.csv')
test_X = test_dataset.drop(columns=['Id'])
test_X[numerical_features] = imputer.transform(test_X[numerical_features])
test_X[categorical_features] = cat_imputer.transform(test_X[categorical_features])
test_X = ct.transform(test_X).toarray()
test_X = sc.transform(test_X)

test_y = best_regressor.predict(test_X)

submission = pd.DataFrame({'Id': test_dataset['Id'], 'SalePrice': test_y})
submission.to_csv('submission.csv', index=False)
