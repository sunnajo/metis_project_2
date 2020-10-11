'''
This module contains functions for partitioning data and cross-validating
regression models
'''
# Import modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold

import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf

# Functions
def train_test_cv(df, cols, target):
    '''
    Partitions specified data into train and test sets (80%/20%)
    Converts resulting dataframes into numpy arrays

    Parameters
    ----------
    cols: list of column names for features/independent variables from dataframe
    target: column name for target/dependent variable from dataframe

    Returns
    ----------
    X: numpy array of features/independent variables from training set
    X_test: dataframe of features/independent variables from test set
    y: numpy array of target/dependent variable from training set
    y_test: series of target/dependent variable from test set
    '''
    X, y = df[cols], df[target]

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=20)    # default test_size, random_state
    
    X, y = np.array(X), np.array(y)


def lr_cv(X, y):
    '''
    Performs KFold cross-validation on linear regression model and computes model metrics

    Parameters
    ----------
    X: numpy array of features/independent variables from training set
    y: numpy array of target/dependent variable from training set

    Returns
    ----------
    Model metrics with cross-validation: R^2 score for training and validation sets
    with each fold, mean R^2 score for validation set across all folds;
    mean training/validation score ratios, MAE, RMSE across all folds
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=71)   # default values

    r2s_train = []
    r2s_val = []
    ratios = []
    maes = []
    rmses = []

    for train_ind, val_ind in kf.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        lm = LinearRegression()
        lm.fit(X_train, y_train)
    
        y_pred = lm.predict(X_val)
    
        r2s_train.append(lm.score(X_train, y_train))
        r2s_val.append(lm.score(X_val, y_val))
        ratios.append(lm.score(X_train, y_train) / lm.score(X_val, y_val))
        maes.append(mean_absolute_error(y_val, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    print('Linear regression train R^2: ', r2s_train)
    print('Linear regression val R^2: ', r2s_val)
    print(f'Linear regression mean val R^2: {np.mean(r2s_val):.3f} +- {np.std(r2s_val):.3f}')
    print(f'Mean train/val R^2 ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}')
    print('Mean MAE: ', np.mean(maes))
    print('Mean RMSE: ', np.mean(rmses))

def poly_cv(X, y):
    '''
    Performs KFold cross-validation on polynomial regression model and computes model metrics

    Parameters
    ----------
    X: numpy array of features/independent variables from training set
    y: numpy array of target/dependent variable from training set

    Returns
    ----------
    Model metrics with cross-validation: R^2 score for training and validation sets
    with each fold, mean R^2 score for validation set across all folds;
    mean training/validation score ratios, MAE, RMSE across all folds
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=71)   # default values
    
    r2s_train = []
    r2s_val = []
    ratios = []
    maes = []
    rmses = []

    for train_ind, val_ind in kf.split(X, y):
    
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind] 

        # Polynomial transformation of features
        poly = PolynomialFeatures(degree=2)     # default # of degrees
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        X_test_poly = poly.transform(X_test)

        lm_poly = LinearRegression()
        lm_poly.fit(X_train_poly, y_train)
        
        r2s_train.append(lm_poly.score(X_train_poly, y_train))
        r2s_val.append(lm_poly.score(X_val_poly, y_val))
        ratios.append(lm_poly.score(X_train_poly, y_train)/lm_poly.score(X_val_poly, y_val))
        maes.append(mean_absolute_error(y_val, lm_poly.predict(X_val_poly)))
        mses.append(mean_squared_error(y_val, lm_poly.predict(X_val_poly)))

    print('Polynomial regression train R^2:', r2s_train)
    print('Polynomial regression val R^2: ', r2s_val)
    print(f'Polynomial regression mean val R^2: {np.mean(r2s_val):.3f} +- {np.std(r2s_val):.3f}')
    print(f'Mean train/val R^2 ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}', '\n')
    print('Mean MAE: ', np.mean(maes))
    print('Mean RMSE: ', np.mean(rmses))


def lr_ridge_cv(X, y):
    '''
    Performs KFold cross-validation on linear regression model with ridge regularization
    and computes model metrics

    Parameters
    ----------
    X: numpy array of features/independent variables from training set
    y: numpy array of target/dependent variable from training set

    Returns
    ----------
    Model metrics with cross-validation: R^2 score for training and validation sets
    with each fold, mean R^2 score for validation set across all folds;
    mean training/validation score ratios, MAE, RMSE across all folds
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=71)   # default values

    r2s_train = []
    r2s_val = []
    ratios = []
    maes = []
    rmses = []

    for train_ind, val_ind in kf.split(X, y):
    
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind] 

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        lm_reg = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000])  # range of alphas
        lm_reg.fit(X_train_scaled, y_train)
        
        y_pred = lm_reg.predict(X_val_scaled)
        
        r2s_train.append(lm_reg.score(X_train_scaled, y_train))
        r2s_val.append(lm_reg.score(X_val_scaled, y_val))
        ratios.append(lm.score(X_train, y_train) / lm.score(X_val, y_val))
        maes.append(mean_absolute_error(y_val, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    print('Ridge linear regression train R^2: ', r2s_train)
    print('Ridge linear regression val R^2: ', r2s_val, '\n')
    print(f'Ridge linear regression mean val R^2: {np.mean(r2s_val):.3f} +- {np.std(r2s_val):.3f}', '\n')
    print(f'Mean train/val R^2 ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}')
    print('Mean MAE: ', np.mean(maes))
    print('Mean RMSE: ', np.mean(rmses))
    print('alpha: ', lasso_reg.alpha_)


def lr_lasso_cv(X, y):
    '''
    Performs KFold cross-validation on linear regression model with lasso regularization
    and computes model metrics

    Parameters
    ----------
    X: numpy array of features/independent variables from training set
    y: numpy array of target/dependent variable from training set

    Returns
    ----------
    Model metrics with cross-validation: R^2 score for training and validation sets
    with each fold, mean R^2 score for validation set across all folds;
    mean training/validation score ratios, MAE, RMSE across all folds; optimal alpha value
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=71)   # default values

    r2s_train = []
    r2s_val = []
    ratios = []
    maes = []
    rmses = []

    for train_ind, val_ind in kf.split(X, y):
    
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind] 
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        alphavec = 10**np.linspace(-2,2,200)    # range of alpha values
        lasso_reg = LassoCV(alphas=alphavec, cv=5)
        lasso_reg.fit(X_train_scaled, y_train)

        r2_train.append(lasso_reg.score(X_train_scaled, y_train))
        r2_val.append(lasso_reg.score(X_val_scaled, y_val))
        mse.append(mean_squared_error(y_val, lasso_reg.predict(X_val_scaled)))
        ratios.append(lasso_reg.score(X_train_scaled, y_train)/lasso_reg.score(X_val_scaled, y_val))

    print('Lasso linear regression train R^2: ', r2_train)
    print('Lasso linear regression val R^2: ', r2_val)
    print(f'Lasso linear regression mean val R^2: {np.mean(r2_val):.3f} +- {np.std(r2_val):.3f}')
    print(f'Mean train/val R^2 ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}')
    print('Mean MAE: ', np.mean(maes))
    print('Mean RMSE: ', np.mean(rmses))
    print('alpha: ', lasso_reg.alpha_)


def poly_ridge_cv(X, y):
    '''
    Performs KFold cross-validation on polynomial regression model with ridge regularization
    and computes model metrics

    Parameters
    ----------
    X: numpy array of features/independent variables from training set
    y: numpy array of target/dependent variable from training set

    Returns
    ----------
    Model metrics with cross-validation: R^2 score for training and validation sets
    with each fold, mean R^2 score for validation set across all folds;
    mean training/validation score ratios, MAE, RMSE across all folds; optimal alpha value
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=71)   # default values

    r2s_train = []
    r2s_val = []
    ratios = []
    maes = []
    rmses = []

    for train_ind, val_ind in kf.split(X, y):
        
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind] 

        # Polynomial transformation of features
        poly = PolynomialFeatures(degree=2)     # default # of degrees
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        X_test_poly = poly.transform(X_test)
        
        # Scale polynomial features
        scaler = StandardScaler(with_mean=False)
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_val_poly_scaled = scaler.transform(X_val_poly)
        
        lm_poly_reg = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000])     # range of alpha values
        lm_poly_reg.fit(X_train_poly_scaled, y_train)
        
        r2s_train.append(lm_poly_reg.score(X_train_poly_scaled, y_train))
        r2s_val.append(lm_poly_reg.score(X_val_poly_scaled, y_val))
        ratios.append(lm_poly_reg.score(X_train_poly_scaled, y_train)/lm_poly_reg.score(X_val_poly_scaled, y_val))
        maes.append(mean_absolute_error(y_val, lm_poly_reg.predict(X_val_poly_scaled))))
        mses.append(mean_squared_error(y_val, lm_poly_reg.predict(X_val_poly_scaled)))

    print('Ridge polynomial regression train R^2:', r2s_train)
    print('Ridge polynomial regression val R^2: ', r2s_val)
    print(f'Ridge polynomial regression mean val R^2: {np.mean(r2s_val):.3f} +- {np.std(r2s_val):.3f}')
    print(f'Mean train/val R ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}')
    print('Mean MAE: ', np.mean(maes))
    print('Mean RMSE: ', np.mean(np.sqrt(mse)))
    print('alpha: ', lm_poly_reg.alpha_)


def poly_lasso_cv(X, y):
    '''
    Performs KFold cross-validation on polynomial regression model with lasso regularization
    and computes model metrics

    Parameters
    ----------
    X: numpy array of features/independent variables from training set
    y: numpy array of target/dependent variable from training set

    Returns
    ----------
    Model metrics with cross-validation: R^2 score for training and validation sets
    with each fold, mean R^2 score for validation set across all folds;
    mean training/validation score ratios, MAE, RMSE across all folds; optimal alpha value
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=71)   # default values

    r2s_train = []
    r2s_val = []
    ratios = []
    maes = []
    rmses = []

    for train_ind, val_ind in kf.split(X, y):
    
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind] 

        # Polynomial transformation of features
        poly = PolynomialFeatures(degree=2)     # default # of degrees

        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        X_test_poly = poly.transform(X_test)
        
        # Scale polynomial features
        scaler = StandardScaler()

        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_val_poly_scaled = scaler.transform(X_val_poly)

        alphavec = 10**np.linspace(-2,2,200)    # range of alpha values
        lasso_reg = LassoCV(alphas=alphavec, cv=5)
        lasso_reg.fit(X_train_poly_scaled, y_train)

        r2_train.append(lasso_reg.score(X_train_poly_scaled, y_train))
        r2_val.append(lasso_reg.score(X_val_poly_scaled, y_val))
        ratios.append(lasso_reg.score(X_train_poly_scaled, y_train)/lasso_reg.score(X_val_poly_scaled, y_val))
        maes.append(mean_absolute_error(y_val, lasso_reg.predict(X_val_poly_scaled)))
        rmses.append(np.sqrt(mean_squared_error(y_val, lasso_reg.predict(X_val_poly_scaled)))

    print('Lasso polynomial regression train r^2: ', r2_train)
    print('Lasso polynomial regression val r^2: ', r2_val)
    print(f'Lasso polynomial regression mean val r^2: {np.mean(r2_val):.3f} +- {np.std(r2_val):.3f}')
    print(f'Mean train/val R^2 ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}', '\n')
    print('Mean MAE': , np.mean(maes))
    print('Mean RMSE ', np.mean(rmses))
    print('alpha: ', lasso_reg.alpha_)