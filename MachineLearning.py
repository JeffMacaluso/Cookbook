import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("NumPy: ", np.__version__)
print("Pandas: ", pd.__version__)

# Formatting for seaborn plots
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

# Displays all dataframe columns
pd.set_option('display.max_columns', None)

%matplotlib inline

#################################################################################################################
### Checking Missing Values
import missingno as msno  # Visualizes missing values
msno.matrix(df)
msno.heatmap(df)  # Co-occurrence of missing values

#################################################################################################################
### Quick EDA report on dataframe
import pandas_profiling
profile = pandas_profiling.ProfileReport(df)
profile.get_rejected_variables(threshold=0.9)  # Rejected variables w/ high correlation
profile.to_file(outputfile="/tmp/myoutputfile.html")  # Saving report as a file

#################################################################################################################
### Preprocessing
# One-hot encoding multiple columns
df_encoded = pd.get_dummies(df, columns=['a', 'b', 'c'], drop_first=True)
#################################################################################################################
# Normalizing
from sklearn import preprocessing
X_norm = preprocessing.normalize(X, norm='max', axis=1)  # Normalizing across columns

### Cross Validation
# Holdout method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)

# K-fold cross validation
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=46)
cross_val_score(model, X, y, cv=k_fold, n_jobs=-1)

#################################################################################################################
### Probability Threshold Search - xgboost
cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=46)

# Making a dataframe to store results of various iterations
xgbResults = pd.DataFrame(columns=['probabilityThreshold', 'f1'])
accuracy, precision, recall, f1 = [], [], [], []

# Parameters for the model
num_rounds = 8000
params = {'booster': 'gbtree', 'max_depth': 4, 'eta': 0.001, 'objective': "binary:logistic"}

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

print("The Model performace is:")
print(xgbResults.mean())

#################################################################################################################
### Probability Threshold Search - scikit-learn
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
### Grid search
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X, y)
print(clf.best_estimator_, '\n', 
      clf.best_params_, '\n', 
      clf.best_score_)

#################################################################################################################
### Basic model performance
# Regression
def initial_regression_test(X, y):
    """
    Tests multiple regression models and plots performance for cross-validation with a holdout set
    ---Note: Add models here
    ---To-do: - Create a function to get the score
              - Add multiple loss functions (RMSE, MAE) and R^2
              - Add plot
    """
    # Splitting between testing and training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
    
    def get_score(model):
        """
        Fits the model and returns a series containing the RMSE, MAE, and R^2
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return r2
    
    # Linear regression
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression(n_jobs=-1)
    lmScore = get_score(lm)
    
    # Decision tree
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    dtScore = dt.score(X_test, y_test)
    
    # k-NN
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    knnScore = knn.score(X_test, y_test)
    
    # Support Vector Machine
    from sklearn.svm import SVR
    svm = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    svm.fit(X_train, y_train)
    svmScore = svm.score(X_test, y_test)
    
    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1)
    rf.fit(X_train, y_train)
    rfScore = rf.score(X_test, y_test)
    
    # Gradient Boosted Tree
    from sklearn.ensemble import GradientBoostingRegressor
    gbt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbt.fit(X_train, y_train)
    gbtScore = gbt.score(X_test, y_test)
    
    # MLP Neural Network
    from sklearn.neural_network import MLPRegressor
    nn = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                      learning_rate='constant', learning_rate_init=0.001)
    nn.fit(X_train, y_train)
    nnScore = nn.score(X_test, y_test)
    
    # Putting results into a data frame before plotting
    results = pd.DataFrame({'LinearRegression': lmScore, 'DecisionTree': dtScore,
                            'k-NN': knnScore, 'SVM': svmScore, 'RandomForest': rfScore,
                            'GradientBoosting': gbtScore, 'nnMLP': nnScore}, index=['R^2'])
    return results
    
    
initial_regression_test(X, y)

#################################################################################################################
### Ensemble Model Importance
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
### Plotting residuals
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
