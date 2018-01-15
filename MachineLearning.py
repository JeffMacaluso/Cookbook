import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS: ', sys.platform)
print('Python: ', sys.version)
print('NumPy: ', np.__version__)
print('Pandas: ', pd.__version__)

# Formatting for seaborn plots
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')

# Displays all dataframe columns
pd.set_option('display.max_columns', None)

%matplotlib inline


#################################################################################################################
##### Exploratory Data Analysis

# Quick EDA report on dataframe
import pandas_profiling
profile = pandas_profiling.ProfileReport(df)
profile.get_rejected_variables(threshold=0.9)  # Rejected variables w/ high correlation
profile.to_file(outputfile='/tmp/myoutputfile.html')  # Saving report as a file

#################################################################################################################
##### Missing Values

# Plotting missing values
import missingno as msno  # Visualizes missing values
msno.matrix(df)
msno.heatmap(df)  # Co-occurrence of missing values

# Drop missing values
df.dropna(how='any', thresh=None, inplace=True)  # Also 'all' for how, and thresh is an int

# Filling missing values with columnar means
df.fillna(value=df.mean(), inplace=True)

# Filling missing values with interpolation
df.fillna(method='ffill', inplace=True)  #'backfill' for interpolating the other direction

# Filling missing values with a predictive model
def predict_missing_values(data, column, correlationThresh=0.5, cross_validations=3):
    """
    Fills missing values using a random forest regression on highly correlated columns
    Returns a series of the column with missing values filled
    
    To-do: - Add the option to specify columns to use for predictions
           - Add the option for categorical columns
           - Look into other options for handling missing predictors
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    
    # Multi-threading if the dataset is a size where doing so is beneficial
    if data.shape[0] < 100000:
        num_cores = 1  # Single-thread
    else:
        num_cores = -1  # All available cores
    
    # Instantiating the model
    rfImputer = RandomForestRegressor(n_estimators=100, n_jobs=num_cores)
    
    # Calculating the highly correlated columns to use for the model
    highlyCorrelated = abs(data.corr()[inputColumn]) >= correlationThresh
    highlyCorrelated = data[data.columns[highlyCorrelated]]
    highlyCorrelated = highlyCorrelated.dropna(how='any')  # Drops any missing records
    
    # Creating the X/y objects to use for the
    y = highlyCorrelated[column]
    X = highlyCorrelated.drop(column, axis=1)
    
    # Evaluating the effectiveness of the model
    cvScore = np.mean(cross_val_score(rfImputer, X, y, cv=cross_validations, n_jobs=num_cores))
    print('Cross Validation Score:', cvScore)

    # Fitting the model for predictions
    rfImputer.fit(X, y)
    print('R^2:', rfImputer.score(X, y))
    
    # Re-filtering the dataset down to highly correlated columns
    # Filling NA predictors w/ columnar mean instead of removing
    X_missing = data[highlyCorrelated.columns]
    X_missing = X_missing.drop(column, axis=1)
    X_missing = X_missing.fillna(X_missing.mean())
    
    # Filtering to rows with missing values before generating predictions
    missingIndexes = data[data[column].isnull()].index
    X_missing = X_missing.iloc[missingIndexes]
    
    # Predicting the missing values
    predictions = rfImputer.predict(X_missing)
    
    # Preventing overwriting of original dataframe
    data = data.copy()

    # Looping through the missing values and replacing with predictions
    for i, idx in enumerate(missingIndexes):
        data.set_value(idx, column, predictions[i])
    
    return data[column]
    
    
df[colName] = predict_missing_values(df, colName)

#################################################################################################################
##### Preprocessing

# One-hot encoding multiple columns
df_encoded = pd.get_dummies(df, columns=['a', 'b', 'c'], drop_first=True)

# Converting a categorical column to numbers
df['TargetVariable'].astype('category').cat.codes

# Normalizing
from sklearn import preprocessing
X_norm = preprocessing.normalize(X, norm='max', axis=0)  # Normalizing across columns

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
# Grid search
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X, y)
print(clf.best_estimator_, '\n', 
      clf.best_params_, '\n', 
      clf.best_score_)

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
predicted = model.predict_proba(X_test)[:, 1]
expected = y_test

# Creating an empty dataframe to fill
results = pd.DataFrame(columns=['threshold', 'f1'])

# Looping trhough different probability thresholds
for thresh in np.arange(0, 30000):
    pred_bin = pd.Series(predicted).apply(lambda x: 1 if x > (thresh / 100000) else 0)
    f1 = metrics.f1_score(expected, pred_bin)
    tempResults = {'threshold': (thresh / 100000), 'f1': metrics.f1_score(pred_bin, y_test)}
    results = results.append(tempResults, ignore_index = True)
    
best_index = list(result['f1']).index(max(results['f1']))
print(results.ix[best_index])

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
    plot_results(results.drop('TrainingTime(sec)', axis=1), 'Classification Evaluation Metrics')
        
    return results
    
    
initial_classification_test(X, y)
   
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
    plt.plot(np.arange(0, df_results.max().max()), color='darkorange', linestyle='--')
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
