import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('CPU Cores:', os.cpu_count())
print('Python:', sys.version)
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)

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

# Printing the percentage of missing values per column
def percent_missing(dataframe):
    '''
    Prints the percentage of missing values for each column in a dataframe
    '''
    # Summing the number of missing values per column and then dividing by the total
    sumMissing = dataframe.isnull().values.sum(axis=0)
    pctMissing = sumMissing / dataframe.shape[0]
    
    if sumMissing.sum() == 0:
        print('No missing values')
    else:
        # Looping through and printing out each columns missing value percentage
        print('Percent Missing Values:', '\n')
        for idx, col in enumerate(dataframe.columns):
            if sumMissing[idx] > 0:
                print('{0}: {1:.2f}%'.format(col, pctMissing[idx] * 100))
        

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
    '''
    Fills missing values using a random forest model on highly correlated columns
    Returns a series of the column with missing values filled
    
    To-do: - Add the option to specify columns to use for predictions
           - Look into other options for handling missing predictors
    '''
    from sklearn.model_selection import cross_val_score
    from sklearn import ensemble
    
    # Printing number of percentage values missing
    pctMissing = data[column].isnull().values.sum() / data.shape[0]
    print('Predicting missing values for {0}\n'.format(column))
    print('Percentage missing: {0:.2f}%'.format(pctMissing*100))
    
    # Multi-threading if the dataset is a size where doing so is beneficial
    if data.shape[0] < 100000:
        num_cores = 1  # Single-thread
    else:
        num_cores = -1  # All available cores
    
    # Instantiating the model
    # Picking a classification model if the number of unique labels are 25 or under
    num_unique_values = len(np.unique(data[column]))
    if num_unique_values > 25 or data[column].dtype != 'category':
        print('Variable is continuous')
        rfImputer = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=num_cores)
    else:
        print('Variable is categorical with {0} classes').format(num_unique_values)
        rfImputer = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=num_cores)
    
    # Calculating the highly correlated columns to use for the model
    highlyCorrelated = abs(data.corr()[column]) >= correlationThresh
    
    # Exiting the function if there are not any highly correlated columns found
    if highlyCorrelated.sum() < 2:  # Will always be 1 because of correlation with self
        print('Error: No correlated variables found. Re-try with less a lower correlation threshold')
        return  # Exits the function
    highlyCorrelated = data[data.columns[highlyCorrelated]]
    highlyCorrelated = highlyCorrelated.dropna(how='any')  # Drops any missing records
    print('Using {0} highly correlated features for predictions\n'.format(highlyCorrelated.shape[1]))
    
    # Creating the X/y objects to use for the
    y = highlyCorrelated[column]
    X = highlyCorrelated.drop(column, axis=1)
    
    # Evaluating the effectiveness of the model
    cvScore = np.mean(cross_val_score(rfImputer, X, y, cv=cross_validations, n_jobs=num_cores))
    print('Cross Validation Score:', cvScore)

    # Fitting the model for predictions and displaying initial results
    rfImputer.fit(X, y)
    if num_unique_values > 25 or data[column].dtype.name != 'category':
        print('R^2:', rfImputer.score(X, y))
    else:
        print('Accuracy:', rfImputer.score(X, y))
    
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
##### Outliers

# TODO: - Add docstrings to functions
#       - Add other functions (GESD, local outlier factor, isolation forests, etc.)

# Detecting outliers with Interquartile Range (IQR)
# Note: The function in its current form is taken from Chris Albon's Machine Learning with Python Cookbook
def iqr_indices_of_outliers(X):
    '''
    Detects outliers using the interquartile range (IQR) method
    
    Input: An array of a variable to detect outliers for
    Output: An array with indices of detected outliers
    '''
    q1, q3 = np.percentile(X, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    outlier_indices = np.where((X > upper_bound) | (X < lower_bound))
    return outlier_indices


# Detecting outliers with Z scores
def z_score_indices_of_outliers(X, threshold=3):
    '''
    Detects outliers using the Z score method method
    
    Input: - X: An array of a variable to detect outliers for
           - threshold: The number of standard deviations from the mean
                        to be considered an outlier
                        
    Output: An array with indices of detected outliers
    '''
    X_mean = np.mean(X)
    X_stdev = np.std(X)
    z_scores = [(y - X_mean) / X_stdev for y in X]
    outlier_indices = np.where(np.abs(z_scores) > threshold)
    return outlier_indices


# Detecting outliers with the Elliptical Envelope method
def ellipses_indices_of_outliers(X, contamination=0.1):
    '''
    Detects outliers using the elliptical envelope method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.covariance import EllipticEnvelope
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X[:, non_categorical]  # Subsetting to columns without categorical indexes

    # Testing if there are an adequate number of features
    if X.shape[0] < X.shape[1] ** 2.:
        print('Will not perform well. Reduce the dimensionality and try again.')
        return
    
    # Creating and fitting the detector
    outlier_detector = EllipticEnvelope(contamination=contamination)
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices


# Detecting outliers with the Isolation Forest method
def isolator_forest_indices_of_outliers(X, contamination=0.1, n_estimators=100):
    '''
    Detects outliers using the isolation forest method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.ensemble import IsolationForest
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X[:, non_categorical]  # Subsetting to columns without categorical indexes
    
    # Creating and fitting the detector
    outlier_detector = IsolationForest(contamination=contamination, n_estimators=n_estimators)
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices

# Detecting outliers with the One Class SVM method
def one_class_svm_indices_of_outliers(X):
    '''
    Detects outliers using the one class SVM method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.svm import OneClassSVM
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X[:, non_categorical]  # Subsetting to columns without categorical indexes

    # Testing if there are an adequate number of features
    if X.shape[0] < X.shape[1] ** 2.:
        print('Will not perform well. Reduce the dimensionality and try again.')
        return
    
    # Creating and fitting the detector
    outlier_detector = OneClassSVM()
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices


# Detecting all outliers in a dataframe using multiple methods
def outlier_report(dataframe, z_threshold=3, per_threshold=0.95, contamination=0.1, n_trees=100):
    '''
    TODO: - Write Docstring
          - Finish commenting function
    '''
    
    # Converting to a pandas dataframe if it is an array
    if type(dataframe) != 'pandas.core.frame.DataFrame':
        try:
            dataframe = pd.DataFrame(dataframe)
        except:
            return 'Must be either a dataframe or a numpy array'
    
    # Creating a copy to avoid fidelity issues
    dataframe = dataframe.copy()
    
    # Dropping categorical columns
    dataframe = dataframe.select_dtypes(exclude=['bool_'])
    for column in dataframe.columns:
        num_unique_values = len(dataframe[column].unique())
        if num_unique_values < 30:
            dataframe = dataframe.drop(column, axis=1)
    
    # Functions for performing outlier detection
    def iqr_indices_of_outliers(X):
        '''
        Determines outliers with the interquartile range (IQR) method
    
        Input: An array of one variable to detect outliers for
        Output: An array with indices of detected outliers
        '''
        q1, q3 = np.percentile(X, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        outlier_indices = np.where((X > upper_bound) | (X < lower_bound))
        return outlier_indices
    
    def z_score_indices_of_outliers(X):
        '''
        Determines outliers based off of the Z score
    
        Input: An array of one variable to detect outliers for
        Output: An array with indices of detected outliers
        '''
        X_mean = np.mean(X)
        X_stdev = np.std(X)
        z_scores = [(y - X_mean) / X_stdev for y in X]
        outlier_indices = np.where(np.abs(z_scores) > z_threshold)
        return outlier_indices
    
    def percentile_indices_of_outliers(X):
        '''
        Determines outliers based off of percentiles
    
        Input: An array of one variable to detect outliers for
        Output: An array with indices of detected outliers
        '''
        diff = (1 - per_threshold) / 2.0
        minval, maxval = np.percentile(X, [diff, 100 - diff])
        outlier_indices = np.where((X < minval) | (X > maxval))
        return outlier_indices
    
    def ellipses_envelope_indices_of_outliers(X):
        '''
        Detects outliers using the elliptical envelope method
    
        Input: An array of all variables to detect outliers for
        Output: An array with indices of detected outliers
        '''
        from sklearn.covariance import EllipticEnvelope
    
        # Creating and fitting the detector
        outlier_detector = EllipticEnvelope(contamination=contamination)
        outlier_detector.fit(X)
    
        # Predicting outliers and outputting an array with 1 if it is an outlier
        outliers = outlier_detector.predict(X)
        outlier_indices = np.where(outliers == -1)
        return outlier_indices
    
    def isolation_forest_indices_of_outliers(X):
        '''
        Detects outliers using the isolation forest method
    
        Input: An array of all variables to detect outliers for
        Output: An array with indices of detected outliers
        '''
        from sklearn.ensemble import IsolationForest
    
        # Creating and fitting the detector
        outlier_detector = IsolationForest(n_estimators=n_trees,
                                           contamination=contamination)
        outlier_detector.fit(X)
    
        # Predicting outliers and outputting an array with 1 if it is an outlier
        outliers = outlier_detector.predict(X)
        outlier_indices = np.where(outliers == -1)
        return outlier_indices
    
    def one_class_svm_indices_of_outliers(X):
        '''
        Detects outliers using the one class SVM method
    
        Input: An array of all variables to detect outliers for
        Output: An array with indices of detected outliers
        '''
        from sklearn.svm import OneClassSVM
    
        # Creating and fitting the detector
        outlier_detector = OneClassSVM()
        outlier_detector.fit(X)
    
        # Predicting outliers and outputting an array with 1 if it is an outlier
        outliers = outlier_detector.predict(X)
        outlier_indices = np.where(outliers == -1)
        return outlier_indices
    
    
    # Dictionaries for individual features to be packaged into a master dictionary
    iqr_outlier_indices = {}
    z_score_outlier_indices = {}
    percentile_outlier_indices = {}
    multiple_outlier_indices = {}  # Indices with two or more detections
    
    print('Detecting outliers', '\n')
    
    # Creating an empty data frame to fill with results
    results = pd.DataFrame(columns=['IQR', 'Z Score', 'Percentile', 'Multiple'])
    
    # Single column outlier tests
    print('Single feature outlier tests')
    for feature in range(dataframe.shape[1]):
        
        # Gathering feature names for use in output dictionary and results dataframe
        feature_name = dataframe.columns[feature]
        
        # Finding outliers
        iqr_outliers = iqr_indices_of_outliers(dataframe.iloc[:, feature])[0]
        z_score_outliers = z_score_indices_of_outliers(dataframe.iloc[:, feature])[0]
        percentile_outliers = percentile_indices_of_outliers(dataframe.iloc[:, feature])[0]
        multiple_outliers = np.intersect1d(iqr_outliers, z_score_outliers)  # TODO: Fix this
        
        # Adding to the empty dictionaries
        iqr_outlier_indices[feature_name] = iqr_outliers
        z_score_outlier_indices[feature_name] = z_score_outliers
        percentile_outlier_indices[feature_name] = percentile_outliers
        multiple_outlier_indices[feature_name] = multiple_outliers
        
        # Adding to results dataframe
        outlier_counts = {'IQR': len(iqr_outliers),
                          'Z Score': len(z_score_outliers),
                          'Percentile': len(percentile_outliers),
                          'Multiple': len(multiple_outliers)}
        outlier_counts_series = pd.Series(outlier_counts, name=feature_name)
        results = results.append(outlier_counts_series)
    
    # Calculating the subtotal of outliers found
    results_subtotal = results.sum()
    results_subtotal.name = 'Total'
    results = results.append(results_subtotal)
    
    # Calculating the percent of total values in each column
    num_observations = dataframe.shape[0]
    results['IQR %'] = results['IQR'] / num_observations
    results['Z Score %'] = results['Z Score'] / num_observations
    results['Percentile %'] = results['Percentile'] / num_observations
    results['Multiple %'] = results['Multiple'] / num_observations
    
    # Printing the results dataframe as a table
    print(results, '\n')
    
    # All column outlier tests
    print('All feature outlier tests')
    ellipses_envelope_outlier_indices = ellipses_envelope_indices_of_outliers(dataframe)
    print('- Ellipses Envelope: {0}'.format(len(ellipses_envelope_outlier_indices[0])))
    
    isolation_forest_outlier_indices = isolation_forest_indices_of_outliers(dataframe)
    print('- Isolation Forest: {0}'.format(len(isolation_forest_outlier_indices[0])))

    one_class_svm_outlier_indices = one_class_svm_indices_of_outliers(dataframe)
    print('- One Class SVM: {0}'.format(len(one_class_svm_outlier_indices[0])))

    # Putting together the final dictionary for output
    all_outlier_indices = {}
    all_outlier_indices['Ellipses Envelope'] = ellipses_envelope_outlier_indices
    all_outlier_indices['Isolation Forest'] = isolation_forest_outlier_indices
    all_outlier_indices['One Class SVM'] = one_class_svm_outlier_indices
    all_outlier_indices['IQR'] = iqr_outlier_indices
    all_outlier_indices['Z Score'] = z_score_outlier_indices
    all_outlier_indices['Percentile'] = percentile_outlier_indices
    all_outlier_indices['Multiple'] = multiple_outlier_indices
    
    return all_outlier_indices

        
outlier_report(df)['feature']['Outlier type']  # Returns array of indices for outliers
# or
outlier_report(df)['Multiple feature outlier type']  # Returns array of indices for outliers
   
#################################################################################################################
##### Preprocessing

# One-hot encoding multiple columns
df_encoded = pd.get_dummies(df, columns=['a', 'b', 'c'], drop_first=True)

# Converting a categorical column to numbers
df['TargetVariable'].astype('category').cat.codes

# Scaling from 0 to 1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)  # Normalizing across columns

# Principal Component Analysis (PCA)
def fit_PCA(X, num_components=0.99):
    '''
    Performs min-max normalization and PCA transformation on the input data array
    
    Inputs:
        - X: An array of values to perform PCA on
        - num_components: If >1, the number of principal components desired
                          If <1, the percentage of variance explained desired
    
    Outputs:
        - An array of the principal components
        
    TODO: Add check if data is already normalized
    '''
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    # Checking if the input is a numpy array and converting it if not
    if type(X) != np.ndarray:
        X = np.array(X)
    
    # Normalizing data before PCA
    min_max_scaler = preprocessing.MinMaxScaler()
    X_norm = min_max_scaler.fit_transform(X)
    
    # Performing PCA
    pca = PCA(n_components=num_components)
    pca.fit(X_norm)
    
    # Reporting explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    print('Total variance % explained:', sum(explained_variance))
    print()
    print('Variance % explained by principal component:')
    for principal_component in range(len(explained_variance)):
        print(principal_component, ':', explained_variance[principal_component])
        
    # Transforming the data before returning
    principal_components = pca.transform(X_norm)
    return principal_components
 
    
# Oversampling
def oversample_binary_label(dataframe, label_column):
    '''
    TODO: 
        - Write docstring
        - Make this work with multiple classes
    '''
    
    # Counting the classes
    class_0_count, class_1_count = dataframe[label_column].value_counts()
    
    # Creating two dataframes for each class
    dataframe_class_0 = dataframe[dataframe[label_column] == dataframe[label_column].unique()[0]]
    dataframe_class_1 = dataframe[dataframe[label_column] == dataframe[label_column].unique()[1]]
    
    # Oversampling
    # TODO: Rename this
    # TODO: Make this dynamically work w/ the smaller class
    dataframe_class_1_oversampled = dataframe_class_1.sample(class_0_count, replace=True)
    dataframe_oversampled = pd.concat([dataframe_class_0, dataframe_class_1_oversampled], axis=0)
    
    # Printing results
    print('Oversampled number of observations in each class:')
    print(dataframe_oversampled[label_column].value_counts())
    
    return dataframe_oversampled
