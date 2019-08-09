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
##### Cross Validation

# Holdout method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)

# K-fold cross validation
from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=46)
cross_val_score(model, X, y, cv=k_fold, n_jobs=-1)

#################################################################################################################
##### Hyperparameter tuning

# Random Search
def hyperparameter_random_search(X, y, model=None, parameters=None, num_folds=5, num_iterations=50):
    '''
    Performs a random search on hyperparameters and 
    '''
    # Randomized Search
    from sklearn.model_selection import RandomizedSearchCV

    # Providing a parameters for a random forest if parameters are not specified
    if parameters is None:
        from scipy.stats import randint as sp_randint

        # Specifying parameters and distributions to sample from
        parameters = {'max_depth': [3, None],
                      'max_features': sp_randint(1, X.shape[1]),
                      'min_samples_split': sp_randint(2, 11),
                      'min_samples_leaf': sp_randint(1, 11),
                      'bootstrap': [True, False],
                      'criterion': ['gini', 'entropy']}

    # Instantiating a model if it isn't provided to the function
    if model is None:
        # Picking between a classifier or regressor based on the number of unique labels
        if len(np.unique(y)) < 50:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_jobs=-1)

    # Performing randomized search
    model = RandomizedSearchCV(model, param_distributions=parameters,
                               n_iter=num_iterations, n_jobs=-1, cv=num_folds)
    model.fit(X, y)

    # Reporting the results
    print('Best Estimator:', model.best_estimator_)
    print('Best Parameters:', model.best_params_)
    print('Best Score:', model.best_score_)
    
    return model

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

# Iteratively training ensemble models
def iteratively_train_random_forest(model, num_trees_to_try, X_train, y_train, X_test, y_test):
    '''
    TODO: 
        - Write docstring
        - Allow different metrics
        - Adjust for regression vs. classification
    '''
    # Enforcing the model has a warm start for iterative training
    if model.warm_start == False:
        model.set_params(warm_start=True)
    
    # Adding a seed if it does not exist
    if model.random_state == None:
        model.set_params(random_state=46)
        
    # Collecting the error for plotting
    errors = []
    
    # Iteratively training the model
    for num_trees in num_trees_to_try:
        model.set_params(n_estimators=num_trees)
        print('Fitting with {0} trees'.format(num_trees))
        model.fit(X_train, y_train)
        errors.append(metrics.log_loss(y_test, growing_rf.predict_proba(X_test)))
    
    # Plotting the error by the number of trees
    plt.figure(figsize=(9, 5))
    plt.plot(num_trees_to_try, errors, '-r')
    plt.title('Error by Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')

#################################################################################################################
##### Class Probability Cutoffs

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
    '''
    Finds the optimal probability cutoff to maximize the F1 score
    Returns the optimal probability cutoff, F1 score, and a plot of the results
    '''
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
##### Prediction Intervals

# Prediction Intervals - Ensemble Scikit-Learn Models
def ensemble_prediction_intervals(model, X, X_train=None, y_train=None, percentile=0.95):
    '''
    Calculates the specified prediction intervals for each prediction
    from an ensemble scikit-learn model.
    
    Inputs:
        - model: The scikit-learn model to create prediction intervals for. This must be
                 either a RandomForestRegressor or GradientBoostingRegressor
        - X: The input array to create predictions & prediction intervals for
        - X_train: The training features for the gradient boosted trees
        - y_train: The training label for the gradient boosted trees
        - percentile: The prediction interval percentile. Default of 0.95 is 0.025 - 0.975
    
    Note: Use X_train and y_train when using a gradient boosted regressor because a copy of
          the model will be re-trained with quantile loss.
          These are not needed for a random forest regressor
    
    Output: A dataframe with the predictions and prediction intervals for X
    
    TO-DO: 
      - Try to optimize by removing loops where possible
      - Fix upper prediction intervals for gradient boosted regressors
      - Add xgboost
    '''
    # Checking if the model has the estimators_ attribute
    if 'estimators_' not in dir(model):
        print('Not an ensemble model - exiting function')
        return

    # Accumulating lower and upper prediction intervals
    lower_PI = []
    upper_PI = []
    
    # Generating predictions to be returned with prediction intervals
    print('Generating predictions with the model')
    predictions = model.predict(X)
    
    # Prediction intervals for a random forest regressor
    # Taken from https://blog.datadive.net/prediction-intervals-for-random-forests/
    if str(type(model)) == "<class 'sklearn.ensemble.forest.RandomForestRegressor'>":
        print('Generating upper and lower prediction intervals')
        
        # Looping through individual records for predictions
        for record in range(len(X)):
            estimator_predictions = []
        
            # Looping through estimators and gathering predictions
            for estimator in model.estimators_:
                estimator_predictions.append(estimator.predict(X[record].reshape(1, -1))[0])
            
            # Adding prediction intervals
            lower_PI.append(np.percentile(estimator_predictions, (1 - percentile) / 2.))
            upper_PI.append(np.percentile(estimator_predictions, 100 - (1 - percentile) / 2.))
    
    # Prediction intervals for gradient boosted trees
    # Taken from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
    if str(type(model)) == "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>":
        # Cloning the model so the original version isn't overwritten
        from sklearn.base import clone
        quantile_model = clone(model)
        
        # Calculating buffer for upper/lower alpha to get the Xth percentile
        alpha_buffer = ((1 - x) / 2)
        alpha = percentile + alpha_buffer
        
        # Setting the loss function to quantile before re-fitting
        quantile_model.set_params(loss='quantile')
        
        # Upper prediction interval
        print('Generating upper prediction intervals')
        quantile_model.set_params(alpha=alpha)
        quantile_model.fit(X_train, y_train)
        upper_PI = quantile_model.predict(X)
        
        # Lower prediction interval
        print('Generating lower prediction intervals')
        quantile_model.set_params(alpha=(1 - alpha))
        quantile_model.fit(X_train, y_train)
        lower_PI = quantile_model.predict(X)
    
    # Compiling results of prediction intervals and the actual predictions
    results = pd.DataFrame({'lower_PI': lower_PI,
                            'prediction': predictions,
                            'upper_PI': upper_PI})
    
    return results


#################################################################################################################
##### Ensemble Predictions
  
# Blending predictions - xgboost
def blend_xgboost_predictions(train_features, train_labels, prediction_features, num_models=3):
    '''
    Trains the number of specified xgboost models and averages the predictions
    
    Inputs: 
        - train_features: A numpy array of the features for the training dataset
        - train_labels: A numpy array of the labels for the training dataset
        - prediction_features: A numpy array of the features to create predictions for
        - num_models: The number of models to train
        
    Outputs:
        - A numpy array of point or class probability predictions
    '''
    
    # Auto-detecting if it's a classification problem and setting the objective for the model
    # Adjust the num_classes cutoff if dealing with a high number of classes
    num_classes = len(np.unique(train_labels))
    if num_classes < 50:
        is_classification = 1
        if num_classes == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softprob'
    else:
        is_classification = 0
        objective = 'reg:linear'
        
    # Creating the prediction object to append results to
    predictions = []
    
    # Parameters for the model - http://xgboost.readthedocs.io/en/latest/parameter.html
    num_rounds = 100
    params = {'booster': 'gbtree',
              'max_depth': 6,  # Default is 6
              'eta': 0.3,  # Step size shrinkage. Default is 0.3
              'alpha': 0,  # L1 regularization. Default is 0.
              'lambda': 1,  # L2 regularization. Default is 1.
              
              # Use reg:linear for regression
              # Use binary:logistic, or multi:softprob for classification
              # Add gpu: to the beginning if training with a GPU. Ex. 'gpu:'+objective
              'objective': objective
             }
    
    # Adding the required parameter for num_classes if performing multiclass classificaiton
    if is_classification == 1 and num_classes != 2:
        params['num_class'] = num_classes
    
    # Creating DMatrix objects from X/y
    D_train = xgb.DMatrix(train_features, label=train_labels)
    D_test = xgb.DMatrix(prediction_features)
    
    # Training each model and gathering the predictions
    for num_model in range(num_models):
        
        # Progress printing for every 10% of completion
        if (num_model+1) % (round(num_models) / 10) == 0:
            print('Training model number', num_model+1)
        
        # Training the model and gathering predictions
        model = xgb.train(params, D_train, num_rounds)
        model_prediction = model.predict(D_test)
        predictions.append(model_prediction)
    
    # Averaging the predictions for output
    predictions = np.asarray(predictions).mean(axis=0)
    
    return predictions
  
# Blending predictions - Scikit-Learn
def blend_sklearn_predictions(model, train_features, train_labels, prediction_features, num_models=3):
    '''
    Trains the number of specified scikit-learn models and averages the predictions
    
    Inputs: 
        - train_features: A numpy array of the features for the training dataset
        - train_labels: A numpy array of the labels for the training dataset
        - prediction_features: A numpy array of the features to create predictions for
        - num_models: The number of models to train
        
    Outputs:
        - A numpy array of point or class probability predictions
    '''
    from sklearn.base import clone
    
    # Auto-detecting if it's a classification problem
    # Adjust the num_classes cutoff if dealing with a high number of classes
    num_classes = len(np.unique(train_labels))
    if num_classes < 50:
        is_classification = 1
    else:
        is_classification = 0
        
    # Creating the prediction object to append results to
    predictions = []
        
    # Training each model and gathering the predictions
    for num_model in range(num_models):
        
        # Progress printing for every 10% of completion
        if (num_model+1) % (round(num_models) / 10) == 0:
            print('Training model number', num_model+1)
        
        # Cloning the original model
        model_iteration = clone(model)
        
        # Training the model
        model_iteration.fit(train_features, train_labels)
        
        # Gathering predictions
        if is_classification == 1:
            model_prediction = model_iteration.predict_proba(prediction_features)
        else:
            model_prediction = model_iteration.predict(prediction_features)
        predictions.append(model_prediction)
    
    # Averaging the predictions for output
    predictions = np.asarray(predictions).mean(axis=0)
    
    return predictions


#################################################################################################################
##### Feature Selection

def stepwise_logistic_regression(X, y, k_fold=True):
    '''
    TODO: 
        - Write docstring
        - Fix IndexError for k_fold
    '''
    # Enforcing numpy arrays to use X[:, feature_index] instead of X.iloc[:, feature_index] for dataframes
    X_feature_selection = np.array(X)
    X_train_feature_selection, X_test_feature_selection, y_train, y_test = train_test_split(
        X_feature_selection, y, test_size=0.30, random_state=46)

    # Creating a list of the number of features to try in the stepwise selection
    num_features_list = np.arange(X.shape[1], 2, -1)

    # List of accuracy to be filled within the for loop
    accuracies = []

    # Testing stepwise feature selection with a different number of features
    for num_features in num_features_list:

        # Printing the progress at different intervals to not spam the output
        if num_features % 5 == 0:
            print('Testing with {0} features'.format(num_features))

        # Using stepwise feature selection to determine which features to select
        estimator = LogisticRegression()
        selector = feature_selection.RFE(estimator, num_features, step=1)
        selector.fit(X_train_feature_selection, y_train)
        feature_indexes = list(selector.get_support(indices=True))
        
        # Training a final model with the specific features selected and gathering the accuracy
        model = LogisticRegression()
        
        # Evaluating with either k-folds cross validation
        if k_fold == True:
            # Constructing new X with the updated features
            X_feature_selection = X_feature_selection[:, feature_indexes]
            k_fold = KFold(n_splits=5, shuffle=True, random_state=46)
            mean_accuracy = cross_val_score(model, X_feature_selection, y, cv=5, n_jobs=-1).mean()
            accuracies.append(mean_accuracy)
            
        # Evaluating with the holdout method
        else:
            # Constructing new X_train/X_test with the updated features
            X_train_feature_selection = X_train_feature_selection[:, feature_indexes]
            X_test_feature_selection = X_test_feature_selection[:, feature_indexes]
            model.fit(X_train_feature_selection, y_train)
            accuracy = model.score(X_test_feature_selection, y_test)
            accuracies.append(accuracy)

    # Putting the results into a data frame and creating a plot of the accuracy by number of features
    feature_selection_results = pd.DataFrame(
        {'Accuracy': accuracies, 'NumFeatures': num_features_list})
    plt.plot(feature_selection_results['NumFeatures'],
             feature_selection_results['Accuracy'])
    plt.title('Accuracy by Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.show()

    # Viewing the output as a sorted data frame
    return feature_selection_results.sort_values('Accuracy', ascending=False)


#################################################################################################################
##### Evaluating Clusters

def evaluate_k_means(data, max_num_clusters=10, is_data_scaled=True):
    '''
    TODO: Write Docstring
    '''
    from sklearn.cluster import KMeans
    
    # Min max scaling the data if it isn't already scaled
    if is_data_scaled == False:
        from sklearn.preprocessing import MinMaxScaler
        data = MinMaxScaler().fit_transform(data)
    
    # For gathering the results and plotting
    inertia = []
    clusters_to_try = np.arange(2, max_num_clusters+1)
    
    # Iterating through the clusters and gathering the inertia
    for num_clusters in np.arange(2, max_num_clusters+1):
        print('Fitting with {0} clusters'.format(num_clusters))
        model = KMeans(n_clusters=num_clusters, n_jobs=-1)
        model.fit(data)
        inertia.append(model.inertia_)
    
    # Plotting the results
    plt.figure(figsize=(10, 7))
    plt.plot(clusters_to_try, inertia, marker='o')
    plt.xticks(clusters_to_try)
    plt.xlabel('# Clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia by Number of Clusters')
    plt.show()
    
    return inertia


#################################################################################################################
##### Saving & Loading Models

def save_model(model, filepath, add_timestamp=True):
    '''
    TODO: Write docstring
    '''
    import os
    
    # Creating the sub directory if it does not exist
    directory = filepath.split('/')[:-1]  # Gathering the components of the file path
    directory = '/'.join(directory) + '/'  # Formatting into the directory/subdirectory/ format
    if not os.path.exists(directory):
        print('Creating the directory')
        os.makedirs(directory)
        
    # Adding the date to the end of the file name if it doesn't exist
    # E.g. instead of model.pkl, model_yyyymmdd.pkl
    if add_timestamp == True:
        import datetime
        today = datetime.datetime.today().strftime('%Y%m%d')
        today = '_' + today + '.'
        filepath = filepath.split('.')
        filepath.insert(1, today)
        filepath = ''.join(filepath)
    
    print('Saving model')
    pickle.dump(model, open(filepath, 'wb'))
    print('Model saved')
