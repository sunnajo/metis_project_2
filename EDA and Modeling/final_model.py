'''
This module contains functions for re-training the training/validation
data sets on the final model and evaluating the model with various metrics
and diagnostic plots
'''

def poly_final(X_train, y_train):
    '''
    Fits and/or transforms features/independent and target variables from
    training and test data sets using PolynomialFeatures
    Instantiates, fits, scores, and evaluates metrics of a linear regression
    model with polynomial features with sklearn
    
    Parameters
    ----------
    X_train: dataframe of features/independent variables from training set
    y_train: series of target/dependent variable from training set
  
    Returns
    ----------
    Model metrics: R^2 score, train/test score ratios, mean absolute error,
    root mean squared error
    '''
    # Transform features prior to testing model
    poly = PolynomialFeatures(degree=2)    # default degree

    X_train_poly = poly.fit_transform(X_train.values)
    X_test_poly = poly.transform(X_test.values)

    # Train model
    lm_poly = LinearRegression()
    lm_poly.fit(X_train_poly, y_train)

    # Evaluate model
    y_pred_poly = lm_poly.predict(X_test_poly)

    r2_train = lm_poly.score(X_train_poly, y_train)
    r2_test = lm_poly.score(X_test_poly, y_test)
    mae = mean_absolute_error(y_test, y_pred_poly)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))

    print('Train R^2: ', r2_train)
    print('Test R^2: ', r2_test)
    print('Train/test R^2 ratio: ', r2_train/r2_test)
    print('MAE: ', mae)
    print('RMSE: ', rmse)


def residual_plots(y_test, y_pred):
    '''
    Plots scatterplot of residuals vs. predicted values from model and
    normal Q-Q plot of residuals to test assumption of homoscedasticity
    and normal distribution of residuals

    Parameters
    ----------
    y_test: series of target/dependent variable from test set
    y_pred: series of predicted values for target/dependent variable from model

    Returns
    ----------
    Scatterplot of residuals vs. predicted values from model
    Normal Q-Q plot of residuals
    '''
    plt.figure(figsize=(20,5))

    plt.subplot(1, 2, 1)
    res = y_test - y_pred
    plt.scatter(y_pred, res)
    plt.axhline(0, color='r')
    plt.title('Residual plot')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')

    plt.subplot(1, 2, 2)
    stats.probplot(res, dist='norm', plot=plt)
    plt.title('Normal Q-Q plot')
    plt.show()


def test_vs_pred(y_test, y_pred):
    '''
    Plots scatterplot of actual vs. predicted values of target/dependent variable
    from model run on test data set to display correlation

    Parameters
    ----------
    y_test: series of target/dependent variable from test set
    y_pred: series of predicted values for target/dependent variable from model run on
    test data

    Returns
    ----------
    Scatterplot of actual vs. predicted values of target/dependent variable from model
    '''
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot(np.linspace(0,100,100), np.linspace(0,100,100), color='r', linestyle='--')
    plt.title('Actual vs. Predictions')
    plt.xlabel('Actual', fontsize=14)
    plt.ylabel('Predictions', fontsize=14)
    plt.show()


def compare_train_test(y_train, y_test, y_pred, X_train, model):
    '''
    Plots scatterplots of actual vs. predicted values of target/dependent variable
    from both training and test data sets to display correlations and assess variability
    in performance on the two sets

    Parameters
    ----------
    y_train: series of target/dependent variable from training set
    y_test: series of target/dependent variable from test set
    y_pred: series of predicted values for target/dependent variable from model run on
    test data
    X_train: dataframe of features/independent variables from training set
    model: model that has been fit on training data

    Returns
    ----------
    Scatterplots of actual vs. predicted values of target/dependent variable from model
    run on both training and test data
    '''
    y_pred_train = model.predict(X_train)

    plt.scatter(y_test, y_pred, alpha=0.7, label='Test')
    plt.scatter(y_train, y_pred_train, alpha=0.3, label='Train')
    plt.plot(np.linspace(0,100,100), np.linspace(0,100,100), color='r', linestyle='--')
    plt.title('Actual vs. Predictions')
    plt.xlabel('Actual', fontsize=14)
    plt.ylabel('Predictions', fontsize=14)
    plt.legend(fontsize=10)
    plt.show()