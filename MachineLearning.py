import sys
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
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
##### Cross Validation

# Holdout method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)

# K-fold cross validation
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=46)
cross_val_score(model, X, y, cv=k_fold, n_jobs=-1)

#################################################################################################################
##### Hyperparameter and model tuning 

### Hyperparameter Tuning
# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

# Specifying parameters and distributions to sample from
param_dist = {'max_depth': [3, None],
              'max_features': sp_randint(1, X.shape[1]),
              'min_samples_split': sp_randint(2, 11),
              'min_samples_leaf': sp_randint(1, 11),
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

randomForest = RandomForestClassifier(n_jobs=-1)

# Performing randomized search
model = RandomizedSearchCV(randomForest, param_distributions=param_dist,
                         n_iter=50, n_jobs=-1, cv=5)
model.fit(X, y)

print('Best Estimator:', model.best_estimator_, '\n', 
      'Best Parameters:', model.best_params_, '\n', 
      'Best Score:', model.best_score_)

# Grid search
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

svc = svm.SVC()

# Performing grid search
model = GridSearchCV(svc, parameters)
model.fit(X, y)

print('Best Estimator:', model.best_estimator_, '\n', 
      'Best Parameters:', model.best_params_, '\n', 
      'Best Score:', model.best_score_)

### Class Probability Cutoffs
# Probability Threshold Search - xgboost
cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=46)

# Making a dataframe to store results of various iterations
xgbResults = pd.DataFrame(columns=['probabilityThreshold', 'f1'])
accuracy, precision, recall, f1 = [], [], [], []

# Parameters for the model
num_rounds = 8000
params = {'booster': 'gbtree', 'max_depth': 4, 'eta': 0.001, 'objective': 'binary:logistic'}

for traincv, testcv in cv:
    
    # Converting the data frames/series to DMatrix objects for xgboost
    Dtrain = xgb.DMatrix(X.ix[traincv], label=y[traincv])
    Dtest = xgb.DMatrix(X.ix[testcv])
    
    # Building the model and outputting class probability estimations
    model = xgb.train(params, Dtrain, num_rounds)
    predictions = model.predict(Dtest)
    temporaryResults = pd.DataFrame(columns=['probabilityThreshold', 'f1'])
    
    # Looping through probability thresholds to gather the f1 score at each threshold
    for probabilityThreshold in np.linspace(0,0.1,100):
        predBin = pd.Series(predictions).apply(lambda x: 1 if x > probabilityThreshold else 0)
        threshF1 = {'probabilityThreshold': probabilityThreshold, 'f1': f1_score(y[testcv], predBin)}
        temporaryResults = temporaryResults.append(threshF1, ignore_index=True)
    
    # Retrieving the f1 score and probability thresholds at the highest f1 score
    bestIndex = list(temporaryResults['f1']).index(max(temporaryResults['f1']))
    bestTempResults = {'probabilityThreshold': temporaryResults.ix[bestIndex][0], 'f1': temporaryResults.ix[bestIndex][1]}
    xgbResults = xgbResults.append(bestTempResults, ignore_index=True)    

print('The Model performace is:')
print(xgbResults.mean())


# Probability Threshold Search - scikit-learn
def optimal_probability_cutoff(model, test_dataset, test_labels, max_thresh=0.3, step_size=0.01):
    """
    Finds the optimal probability cutoff to maximize the F1 score
    Returns the optimal probability cutoff, F1 score, and a plot of the results
    """
    from sklearn import metrics

    # Prediction probabilities of the test dataset
    predicted = model.predict_proba(test_dataset)[:, 1]

    # Creating an empty dataframe to fill with probability cutoff thresholds and f1 scores
    results = pd.DataFrame(columns=['Threshold', 'F1 Score'])

    # Setting f1 score average metric based on binary or multi-class classification
    if len(np.unique(test_labels)) == 2:
        avg = 'binary'
    else:
        avg = 'micro'

    # Looping trhough different probability thresholds
    for thresh in np.arange(0, (max_thresh+step_size), step_size):
        pred_bin = pd.Series(predicted).apply(lambda x: 1 if x > thresh else 0)
        f1 = metrics.f1_score(test_labels, pred_bin, average=avg)
        tempResults = {'Threshold': thresh, 'F1 Score': f1}
        results = results.append(tempResults, ignore_index=True)
        
    # Plotting the F1 score throughout different probability thresholds
    results.plot(x='Threshold', y='F1 Score')
    plt.title('F1 Score by Probability Cutoff Threshold')
    
    best_index = list(results['F1 Score']).index(max(results['F1 Score']))
    print('Threshold for Optimal F1 Score:')
    return results.iloc[best_index]


optimal_probability_cutoff(model, X_test, y_test)
#################################################################################################################
##### Basic model performance testing

# Regression
def initial_regression_test(X, y):
    """
    Tests multiple regression models and gathers performance from cross-validation with a holdout set
    Uses default parameters from sklearn functions for most cases
    
    Outputs: - Dataframe containing RMSE, MAE, and R^2
             - Plots of RMSE/MAE and R^2
    """
    # Splitting between testing and training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
    
    # Min-max scaling for neural nets and SVMs
    from sklearn import preprocessing
    X_train_norm = preprocessing.normalize(X_train, norm='max', axis=0)  # Normalizing across columns
    X_test_norm = preprocessing.normalize(X_test, norm='max', axis=0)  # Normalizing across columns
    
    def get_score(model, norm=False):
        """
        Fits the model and returns a series containing the RMSE, MAE, and R^2
        """
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
        """
        Formats the results and plots them
        """
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
    """
    Tests multiple classification models and gathers performance from cross-validation with a holdout set
    
    Outputs: - Dataframe containing Accuracy, F1, Precision, Recall, Support, Log Loss, and AUC (when applicable)
             - Plots of dataframe contents
    """
    # Splitting between testing and training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
    
    # Min-max scaling for neural nets and SVMs
    from sklearn import preprocessing
    X_train_norm = preprocessing.normalize(X_train, norm='max', axis=0)  # Normalizing across columns
    X_test_norm = preprocessing.normalize(X_test, norm='max', axis=0)  # Normalizing across columns
    
    def get_score(model, norm=False):
        """
        Fits the model and returns a series containing the Accuracy, F1, Precision, Recall, and others
        http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """
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
        """
        Formats the results and plots them
        """
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
    print('\nR^2:', r2)
    print('\nCoefficients')
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
                   the response variable. If not, either a quadratic term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=5)
        
        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        
        
    def multivariate_normal_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the predictors have normal distributions. If they are not normal,
                   a non-linear transformation like a log transformation or box-cox transformation
                   can be performed on the non-normal variable.
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: All variables are multivariate normal')
        print('Using the Anderson-Darling test for normal distribution')
        print('p-values from the test - below 0.05 generally means normality:')
        print()
        non_normal_variables = 0
        
        # Performing the Anderson-Darling test on each variable to test for normality
        for feature in range(features.shape[1]):
            p_value = normal_ad(features[:, feature])[1]
            
            # Adding to total count of non-normality if p-value exceeds threshold
            if p_value > p_value_thresh:
                non_normal_variables += 1
            
            # Printing p-values from the test
            print('{0}: {1}'.format(feature_names[feature], p_value))
                    
        print('\n{0} non-normal variables'.format(non_normal_variables))
        print()

        if non_normal_variables == 0:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
        
        
    def multicollinearity_assumption():
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either removing prepdictors with
                           high Variance Inflation Factor (VIF) values or 
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('\n=======================================================================================')
        print('Assumption 3: Little to no multicollinearity among predictors')
        
        # Plotting the heatmap
        ax = plt.subplot(111)
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr())
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
        else:
            print('Assumption not satisfied')
        
        
    def autocorrelation_assumption():
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a patern that is not explained due to
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
            print('Signs of positive autocorrelation')
            print('\nAssumption not satisfied')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation')
            print('\nAssumption not satisfied')
        else:
            print('Little to no autocorrelation')
            print('\nAssumption satisfied')

            
    def homoscedasticity_assumption():
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('\n=======================================================================================')
        print('Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        
        # Plotting the residuals
        ax = plt.subplot(111)
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show()  
        
        
    linear_assumption()
    multivariate_normal_assumption()
    multicollinearity_assumption()
    autocorrelation_assumption()
    homoscedasticity_assumption()


linear_regression_assumptions(X, y, feature_names=dataset.feature_names)

#################################################################################################################
##### Evaluation Plots

# Residuals
def plot_residuals(model, values, labels):
    """
    Creates two plots: Actual vs. Predicted and Residuals
    """
    # Calculating the predictions and residuals
    predictions = model.predict(values)
    df_results = pd.DataFrame({'Actual': labels, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    # Plotting the actual vs predicted
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=6)

    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()
    
    # Plotting the residuals
    ax = plt.subplot(111)
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()
    
    
plot_residuals(model, X, y)


# Learning Curve
def plot_learning_curve(model, data, labels):
    """
    Plots the learning curve of a model using 3-fold Cross Validation
    """
    from sklearn.model_selection import learning_curve
    learningCurve = learning_curve(model, X, y, cv=3, n_jobs=-1)
    trainScores = learningCurve[1].mean(axis=1)
    testScores = learningCurve[2].mean(axis=1)

    # Putting the results into a dataframe before plotting
    results = pd.DataFrame({'Training Set': trainScores, 'Testing Set': testScores},
                           index=learningCurve[0])  # Training size
    
    # Plotting the curve
    ax = results.plot(figsize=(10, 6), linestyle='-', marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Cross Validation Score')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine    
    plt.legend(loc=(1.04, 0.55))  # Moves the legend outside of the plot
    plt.show()
    

plot_learning_curve(model, X, y)


# Validation Curve
def plot_validation_curve(model, data, labels, param_name, param_values):
    """
    Plots the validation curve of a model using 3-fold Cross Validation
    """
    from sklearn.model_selection import validation_curve
    validationCurve = validation_curve(model, X, y, cv=3, param_name=param_name,
                                       param_range=param_values, n_jobs=-1)

    trainScores = validationCurve[0].mean(axis=1)
    testScores = validationCurve[1].mean(axis=1)

    # Putting the results into a dataframe before plotting
    results = pd.DataFrame({'Training Set': trainScores, 'Testing Set': testScores}, 
                           index=param_values)
    
    # Plotting the curve
    ax = results.plot(figsize=(10, 6), linestyle='-', marker='o')
    plt.title('Validation Curve')
    plt.xlabel(param_name)
    plt.ylabel('Cross Validation Score')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine    
    plt.legend(loc=(1.04, 0.55))  # Moves the legend outside of the plot
    plt.show()
    

param_name = 'n_estimators'
param_range = [10, 30, 100, 300]

plot_validation_curve(model, X, y, param_name, param_range)


# Ensemble Model Importance
def feature_importance(model):
    """
    Plots the feature importance for an ensemble model
    """
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
#################################################################################################################
##### Misc

# Prediction Intervals - Random Forests
def random_forest_prediction_intervals(model, X, percentile=0.95):
    '''
    Calculates the specified prediction intervals for each prediction
    from an ensemble scikit-learn model
    
    Taken from https://blog.datadive.net/prediction-intervals-for-random-forests/
    
    TO-DO: 
      - Try to optimize by removing loops where possible
      - Update to work with gradient boosted trees; see
      http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
    '''
    # Checking if the model has the estimators_ attribute
    if 'estimators_' not in dir(model):
        print('Not an ensemble model - exiting function')
        return

    # Accumulating lower and upper prediction intervals
    lower_PI = []
    upper_PI = []
    
    # Looping through individual records for predictions
    for record in range(len(X)):
        predictions = []
        
        # Looping through estimators and gathering predictions
        for estimator in model.estimators_:
            predictions.append(estimator.predict(X[record].reshape(1, -1))[0])
            
        # Adding prediction intervals
        lower_PI.append(np.percentile(predictions, (1 - percentile) / 2.))
        upper_PI.append(np.percentile(predictions, 100 - (1 - percentile) / 2.))
    
    # Compiling results of prediction intervals and the actual predictions
    predictions = model.predict(X)
    results = pd.DataFrame({'lower_PI': lower_PI,
                            'prediction': predictions,
                            'upper_PI': upper_PI})
    
    return results
