'''
This module contains functions for partitioning data, building and validating linear
regression models, and evaluating models with diagnostic metrics and plots
'''
# Import modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf


# Functions
def train_test(df, cols, target):
    '''
    Partitions specified data into train and test sets (80%/20%)

    Parameters
    ----------
    cols: list of column names for features/independent variables from dataframe
    target: column name for target/dependent variable from dataframe

    Returns
    ----------
    X_train: dataframe of features/independent variables from training set
    X_test: dataframe of features/independent variables from test set
    y_train: series of target/dependent variable from training set
    y_test: series of target/dependent variable from test set
    '''
    X, y = df[cols], df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=20)    # default test_size, random_state


def train_test_val(df, cols, target):
    '''
    Partitions specified data into train, validation, test sets (60%/20%/20%)

    Parameters
    ----------
    cols: list of column names for features/independent variables from dataframe
    target: column name for target/dependent variable from dataframe

    Returns
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    X_test: dataframe of features/independent variables from test set
    y_train: series of target/dependent variable from training set
    y_val: series of target/dependent variable from validation set
    y_test: series of target/dependent variable from test set
    '''
    X, y = df[cols], df[target]

    # Hold out 20% of the data for final testing
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=20)    # default test_size, random_state

    # Further partition data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=11) # default test_size, random_state


def sm_ols(X_train, y_train):
    '''
    Instantiate, fit, and summarize OLS linear regression model with statsmodels

    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    Y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Summary of OLS linear regression model
    '''
    X = X_train
    X = sm.add_constant(X)
    y = y_train
    
    lm = sm.OLS(y, X)
    lm = lm.fit()
    
    return lm.summary()

def linear_regression(X_train, X_val, y_train):
    '''
    Instantiate, fit, score, and evaluate metrics of a linear regression model with sklearn
    
    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    Y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/validation score ratios, mean absolute error,
    root mean squared error
    '''
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    y_pred = lm.predict(X_val)

    print(f'Linear regression val R^2: {lm.score(X_val, y_val):.3f}')
    print('Train/val ratio: ', lm.score(X_train, y_train)/lm.score(X_val, y_val))
    print('MAE: ', mean_absolute_error(y_pred, y_val))
    print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_val)))


def polynomial(X_train, y_train):
    '''
    Fits and/or transforms features/independent and target variables from
    training, validation, and test sets using PolynomialFeatures
    Instantiates, fits, scores, and evaluates metrics of a linear regression
    model with polynomial features with sklearn
    
    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/validation score ratios, mean absolute error,
    root mean squared error
    '''
    # Transform features prior to testing model
    poly = PolynomialFeatures(degree=2)    # default degree

    X_train_poly = poly.fit_transform(X_train.values)
    X_val_poly = poly.transform(X_val.values)
    X_test_poly = poly.transform(X_test.values)

    # Create and validate model
    lm_poly = LinearRegression()
    lm_poly.fit(X_train_poly, y_train)

    y_pred = lm_poly.predict(X_val_poly)

    print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')
    print('Train/val ratio: ', lm_poly.score(X_train_poly, y_train)/lm_poly.score(X_val_poly, y_val))
    print('MAE: ', mean_absolute_error(y_pred, y_val))
    print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_val)))


def lr_ridge(X_train, X_val, y_train):
    '''
    Scales features/independent variables
    Instantiates, fits, scores, and evaluates metrics of a linear regression
    model with ridge regularization with sklearn
    
    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/validation score ratios, mean absolute error,
    root mean squared error
    '''
    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Create and validate model
    lm_reg = Ridge(alpha=1.0)   # default alpha
    lm_reg.fit(X_train_scaled, y_train)

    y_pred = lm_reg.predict(X_val_scaled)

    print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')
    print('Train/val ratio: ', lm_reg.score(X_train_scaled, y_train)/lm_reg.score(X_val_scaled, y_val))
    print('MAE: ', mean_absolute_error(y_pred, y_val))
    print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_val)))


def lr_lasso(X_train, X_val, y_train):
    '''
    Scales features/independent variables
    Instantiates, fits, scores, and evaluates metrics of a linear regression
    model with lasso regularization with sklearn

    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/validation score ratios, mean absolute error,
    root mean squared error
    '''

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Create and validate model
    lasso_model = Lasso(alpha=0.1)  # default alpha
    lasso_model.fit(X_train_scaled, y_train)

    y_pred = lasso_model.predict(X_val)

    print(f'Lasso Regression val R^2: {lasso_model.score(X_val_scaled, y_val):.3f}')
    print('Train/val ratio: ', lasso_model.score(X_train_scaled, y_train)/lasso_model.score(X_val_scaled, y_val))
    print('MAE: ', mean_absolute_error(y_pred, y_val))
    print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_val)))


def poly_ridge(X_train, X_val, y_train):
    '''
    Fits and/or transforms features/independent and target variables from
    training, validation, and test sets using PolynomialFeatures
    Instantiates, fits, scores, and evaluates metrics of a linear regression
    model with polynomial features with ridge regularization with sklearn
    
    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/validation score ratios, mean absolute error,
    root mean squared error
    '''
    # Transform features
    poly = PolynomialFeatures(degree=2)    # default degree

    X_train_poly = poly.fit_transform(X_train.values)
    X_val_poly = poly.transform(X_val.values)
    X_test_poly = poly.transform(X_test.values)

    # Create and validate model
    lm_poly_reg = Ridge(alpha=1.0)  # default alpha
    lm_poly_reg.fit(X_train_poly_scaled, y_train)
    
    y_pred = lm_poly_reg.predict(X_val_poly_scaled)

    print(f'Ridge Regression w/ Polynomial val R^2: {lm_poly_reg.score(X_val_poly_scaled, y_val):.3f}')
    print('Train/val ratio: ', lm_poly_reg.score(X_train_poly_scaled, y_train)/lm_poly_reg.score(X_val_poly_scaled, y_val))
    print('MAE: ', mean_absolute_error(y_pred, y_val))
    print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_val)))


def poly_lasso(X_train, X_val, y_train):
    '''
    Fits and/or transforms features/independent and target variables from
    training, validation, and test sets using PolynomialFeatures
    Instantiates, fits, scores, and evaluates metrics of a linear regression
    model with polynomial features with lasso regularization with sklearn
    
    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    X_val: dataframe of features/independent variables from validation set
    y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/validation score ratios, mean absolute error,
    root mean squared error
    '''
    # Transform features
    poly = PolynomialFeatures(degree=2)    # default degree

    X_train_poly = poly.fit_transform(X_train.values)
    X_val_poly = poly.transform(X_val.values)
    X_test_poly = poly.transform(X_test.values)

    # Create and validate model
    lasso_model = Lasso(alpha=0.1)  # default alpha
    lasso_model.fit(X_train_poly_scaled, y_train)

    y_pred = lasso_model.predict(X_val_poly_scaled)

    print(f'Lasso Polynomial val R^2: {lasso_model.score(X_val_poly_scaled, y_val):.3f}')
    print('Train/val ratio: ', lasso_model.score(X_train_poly_scaled, y_train)/lasso_model.score(X_val_poly_scaled, y_val))
    print('MAE: ', mean_absolute_error(y_pred, y_val))
    print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_val)))
