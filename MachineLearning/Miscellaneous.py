import sys
import os
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('CPU Cores:', os.cpu_count())
print('Python:', sys.version)
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('Scikit-Learn:', sklearn.__version__)

# Formatting for seaborn plots
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')

# Displays all dataframe columns
pd.set_option('display.max_columns', None)

%matplotlib inline

#################################################################################################################
def find_max_qcut_bins(data: np.ndarray, max_bins: int = 25) -> int:
    '''
    Returns the max number of bins for pd.qcut()
    '''
    found_max_bins = False
    while found_max_bins == False:
        try:
            pd.qcut(data, q=max_bins)
            return max_bins
        except:
            max_bins -= 1

#################################################################################################################
##### Assumption Testing
# Linear Regression
def linear_regression_assumptions(features, label, feature_names=None):
    """
    Tests a linear regression on the model to see if assumptions are being met
    """
    from sklearn.linear_model import LinearRegression
    
    # Setting feature names to x1, x2, x3, etc. if they are not defined
    if feature_names is None:
        feature_names = ['X'+str(feature+1) for feature in range(features.shape[1])]
    
    print('Fitting linear regression')
    # Multi-threading if the dataset is a size where doing so is beneficial
    if features.shape[0] < 100000:
        model = LinearRegression(n_jobs=-1)
    else:
        model = LinearRegression()
        
    model.fit(features, label)
    
    # Returning linear regression R^2 and coefficients before performing diagnostics
    r2 = model.score(features, label)
    print()
    print('R^2:', r2, '\n')
    print('Coefficients')
    print('-------------------------------------')
    print('Intercept:', model.intercept_)
    
    for feature in range(len(model.coef_)):
        print('{0}: {1}'.format(feature_names[feature], model.coef_[feature]))

    print('\nPerforming linear regression assumption testing')
    
    # Creating predictions and calculating residuals for assumption tests
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    
    def linear_assumption():
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a polynomial term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)
        
        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        print('If non-linearity is apparent, consider adding a polynomial term')
        
        
    def normal_errors_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
               
        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: The error terms are normally distributed')
        print()
    
        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        p_value = normal_ad(df_results['Residuals'])[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
    
        # Plotting the residuals distribution
        plt.subplots(figsize=(12, 6))
        plt.title('Distribution of Residuals')
        sns.distplot(df_results['Residuals'])
        plt.show()
    
        print()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')
        
        
    def multicollinearity_assumption():
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either remove prepdictors with high
                           Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           
                           This assumption being violated causes issues with interpretability of the 
                           coefficients and the standard errors of the coefficients.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('\n=======================================================================================')
        print('Assumption 3: Little to no multicollinearity among predictors')
        
        # Plotting the heatmap
        plt.figure(figsize = (10,8))
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
        plt.title('Correlation of Variables')
        plt.show()
        
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
       
        # Gathering the VIF for each variable
        VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
        
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
            else:
                print('Assumption possibly satisfied')
                print()
                print('Coefficient interpretability may be problematic')
                print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        else:
            print('Assumption not satisfied')
            print()
            print('Coefficient interpretability will be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        
        
    def autocorrelation_assumption():
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a pattern that is not explained due to
                         the current value being dependent on the previous value.
                         This may be resolved by adding a lag variable of either the dependent
                         variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('\n=======================================================================================')
        print('Assumption 4: No Autocorrelation')
        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        
        durbinWatson = durbin_watson(df_results['Residuals'])
        print('Durbin-Watson:', durbinWatson)
        
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        else:
            print('Little to no autocorrelation', '\n')
            print('Assumption satisfied')

            
    def homoscedasticity_assumption():
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('\n=======================================================================================')
        print('Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        
        # Plotting the residuals
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show() 
        print('If heteroscedasticity is apparent, confidence intervals and predictions will be affected')
        
        
    linear_assumption()
    normal_errors_assumption()
    multicollinearity_assumption()
    autocorrelation_assumption()
    homoscedasticity_assumption()


linear_regression_assumptions(X, y, feature_names=dataset.feature_names)

#################################################################################################################
##### Basic model performance testing

# Regression
def initial_regression_test(X, y):
    '''
    Tests multiple regression models and gathers performance from cross-validation with a holdout set
    Uses default parameters from sklearn functions for most cases
    
    Outputs: - Dataframe containing RMSE, MAE, and R^2
             - Plots of RMSE/MAE and R^2
    '''
    # Splitting between testing and training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
    
    # Min-max scaling for neural nets and SVMs
    from sklearn import preprocessing
    X_train_norm = preprocessing.normalize(X_train, norm='max', axis=0)  # Normalizing across columns
    X_test_norm = preprocessing.normalize(X_test, norm='max', axis=0)  # Normalizing across columns
    
    def get_score(model, norm=False):
        '''
        Fits the model and returns a series containing the RMSE, MAE, and R^2
        '''
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import time

        startTime = time.time()  # Getting training time
        
        # Fits with either regular or normalized training set
        if norm == False:
            model.fit(X_train, y_train)
            totalTime = time.time() - startTime
            predictions = model.predict(X_test)
        
            r2 = model.score(X_test, y_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
        else:
            model.fit(X_train_norm, y_train)
            totalTime = time.time() - startTime
            predictions = model.predict(X_test_norm)
        
            r2 = model.score(X_test_norm, y_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            
        score_results = pd.Series([r2, rmse, mae, totalTime], index=['R^2', 'RMSE', 'MAE', 'TrainingTime(sec)'])
        
        return score_results
    
    
    # Linear regression - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression(n_jobs=-1)
    lmScore = get_score(lm)
    
    # Decision tree - http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    dtScore = get_score(dt)
    
    # k-NN - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knnScore = get_score(knn)
    
    # Support Vector Machine - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    from sklearn.svm import SVR
    svm = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    svmScore = get_score(svm, norm=True)
    
    # Random Forest - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1)
    rfScore = get_score(rf)
    
    # Gradient Boosted Trees - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    from sklearn.ensemble import GradientBoostingRegressor
    gbt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbtScore = get_score(gbt)
    
    # MLP Neural Network - http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    from sklearn.neural_network import MLPRegressor
    nn = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,
                      learning_rate='constant', learning_rate_init=0.001)
    nnScore = get_score(nn, norm=True)
    
    # Putting results into a data frame before plotting
    results = pd.DataFrame({'LinearRegression': lmScore, 'DecisionTree': dtScore,
                            'k-NN': knnScore, 'SVM': svmScore, 'RandomForest': rfScore,
                            'GradientBoosting': gbtScore, 'nnMLP': nnScore})
    
    def plot_results(results, title=None):
        '''
        Formats the results and plots them
        '''
        ax = results.plot(kind='barh')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine    
        plt.legend(loc=(1.04, 0.55))  # Moves the legend outside of the plot
        plt.title(title)
        plt.show()
    
    
    # Plotting the error metrics
    plot_results(results.loc[['RMSE', 'MAE']], 'Error Metrics')
    
    # Plotting the R^2
    plot_results(results.loc['R^2'], '$R^2$')
    
    return results
    
    
initial_regression_test(X, y)


# Classification
def initial_classification_test(X, y):
    '''
    Tests multiple classification models and gathers performance from cross-validation with a holdout set
    
    Outputs: - Dataframe containing Accuracy, F1, Precision, Recall, Support, Log Loss, and AUC (when applicable)
             - Plots of dataframe contents
    '''
    # Splitting between testing and training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
    
    # Min-max scaling for neural nets and SVMs
    from sklearn import preprocessing
    X_train_norm = preprocessing.normalize(X_train, norm='max', axis=0)  # Normalizing across columns
    X_test_norm = preprocessing.normalize(X_test, norm='max', axis=0)  # Normalizing across columns
    
    def get_score(model, norm=False):
        '''
        Fits the model and returns a series containing the Accuracy, F1, Precision, Recall, and others
        http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        '''
        from sklearn.metrics import precision_recall_fscore_support, log_loss, hinge_loss
        import time

        startTime = time.time()  # Getting training time
        
        # Fits with either regular or normalized training set
        if norm == False:
            model.fit(X_train, y_train)
            totalTime = time.time() - startTime
            predictions = model.predict(X_test)
            predictionProbabilities = model.predict_proba(X_test)
        
            # Creating the base metrics for all classification tasks
            accuracy = model.score(X_test, y_test)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average='micro')
            logLoss = log_loss(y_test, predictionProbabilities)
        else:
            model.fit(X_train_norm, y_train)
            totalTime = time.time() - startTime
            predictions = model.predict(X_test_norm)
            predictionProbabilities = model.predict_proba(X_test_norm)
        
            # Creating the base metrics for all classification tasks
            accuracy = model.score(X_test_norm, y_test)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average='micro')
            logLoss = log_loss(y_test, predictionProbabilities)
            
        scoreResults = pd.Series([accuracy, f1, precision, recall, support, logLoss, totalTime],
                                 index=['Accuracy', 'F1', 'Precision', 'Recall', 'Support', 'LogLoss',
                                        'TrainingTime(sec)'])
        
        # Adding additional classification metrics for binary tasks
        if len(np.unique(y)) == 2:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, predictionProbabilities)
            auc = pd.Series(auc, index='AUC')
            scoreResults = scoreResults.append(auc)
        
        return scoreResults
    
    
    # Logistic Regression - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    from sklearn.linear_model import LogisticRegression
    lm = LogisticRegression(C=1.0, penalty='l1', n_jobs=-1)
    lmScore = get_score(lm)
    
    # Decision Tree - http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
    dtScore = get_score(dt)
    
    # k-NN - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knnScore = get_score(knn)
    
    # Support Vector Machine - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    from sklearn.svm import SVC
    svm = SVC(C=1.0, kernel='rbf', probability=True)
    svmScore = get_score(svm, norm=True)
    
    # Random Forest - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
    rfScore = get_score(rf)
    
    # Gradient Boosted Tree - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    from sklearn.ensemble import GradientBoostingClassifier
    gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbtScore = get_score(gbt)
    
    # MLP Neural Network - http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,
                      learning_rate='constant', learning_rate_init=0.001)
    nnScore = get_score(nn, norm=True)
    
    # Putting results into a data frame before plotting
    results = pd.DataFrame({'LogisticRegression': lmScore, 'DecisionTree': dtScore,
                            'k-NN': knnScore, 'SVM': svmScore, 'RandomForest': rfScore,
                            'GradientBoosting': gbtScore, 'nnMLP': nnScore})
    
    def plot_results(results, title=None):
        '''
        Formats the results and plots them
        '''
        ax = results.plot(kind='barh')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine    
        plt.legend(loc=(1.04, 0.55))  # Moves the legend outside of the plot
        plt.title(title)
        plt.show()
    
    
    # Plotting the evaluation metrics
    plot_results(results.drop('TrainingTime(sec)', axis=0), 'Classification Evaluation Metrics')
        
    return results
    
    
initial_classification_test(X, y)
