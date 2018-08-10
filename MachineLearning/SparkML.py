import sys
import time

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('Python:', sys.version)
print('Number of nodes on the cluster:', sc._jsc.sc().getExecutorMemoryStatus().size())

#################################################################################################################
##### Data Preparation
# Vectorizing a training set before feeding into a ML model
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Specifying the name of the column containing the label
labelColumn = 'price'

# Specifying the names of the columns containing the features
featureColumns = ['previous_hour_price']

# Assembling the vectors and outputting the training set
assembler = VectorAssembler(
    inputCols=featureColumns,
    outputCol='features')
output = assembler.transform(df)
trainingDataset = output.select('features', col(labelColumn).alias('label'))

# One Hot Encoding
def one_hot_encode(column, dataframe):
  '''
  Returns a dataframe with an additional one hot encoded column specified on the input
  '''
  from pyspark.ml.feature import OneHotEncoder, StringIndexer
  
  # Indexing the column before one hot encoding
  stringIndexer = StringIndexer(inputCol=column, outputCol='categoryIndex')
  model = stringIndexer.fit(dataframe)
  indexed = model.transform(dataframe)
  
  # One hot encoding the column
  encoder = OneHotEncoder(inputCol='categoryIndex', outputCol=column+'_one_hot')
  encoded = encoder.transform(indexed).drop('categoryIndex')

  return encoded

# Adding a lag variable
def lag_variable(column, dataframe, partition_column, count=12):
  '''
  Returns a dataframe with an additional lag column specified from the input
  '''
  import pyspark.sql.functions as sqlF
  from pyspark.sql.window import Window
  
  lagDF = (dataframe.withColumn('previous_hour_'+column,
                                sqlF.lag(dataframe[column], count=count)
                                    .over(Window.partitionBy()
                                      .orderBy(partition_column))))
  return lagDF

#################################################################################################################
##### Cross Validation
# Train/test split
seed = 46
(trainingDF, testDF) = df.randomSplit([8.0, 2.0], seed)  # 80/20 split

# Evaluating regression
# https://spark.apache.org/docs/2.2.0/ml-tuning.html
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fitting the model
lr = LinearRegression()
model = lr.fit(trainingDF)

regEval = RegressionEvaluator().setLabelCol('label')  # Instantiating & setting the label column
predictDF = model.transform(testDF)  # Generating predictions
testRMSE = regEval.evaluate(predictDF)  # Gathering the RMSE
print('The model had a RMSE on the test set of {0}'.format(testRMSE))


# K-folds
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator().setLabelCol('label')
crossval = CrossValidator(estimator=model,
                          evaluator=regEval,
                          numFolds=3)

# Re-fitting on the entire training set
cvModel = crossval.fit(trainingDF)

# Sliding window for time series model evaluation
def sliding_window_evaluation(dataframe, feature_columns, num_windows=5, test_size=0.2):
    '''
    Takes an input dataframe, splits it into partitions, and performs a sliding window where
    each partition is split between a train/test set and a linear regression is trained
    and evaluated
    
    Meant for analyzing the performance of a time series regression forecasting model as a random
    split is not appropriate in a time series setting
    '''
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Gathering statistics for window partitions and train/test splits
    total_rows = dataframe.count()
    window_size = round(total_rows / num_windows)
    num_training_rows = round((dataframe.count() * (1 - test_size)) / num_windows)

    # Creating a column for partition numbers
    dataframe = (dataframe.withColumn('window_num', ((sqlF.row_number().over(Window.orderBy('date_time_resampled')) - 1) / window_size) + 1)
                          .withColumn('window_num', sqlF.floor(col('window_num'))))  # Truncating to integers
    
    # Specifying the name of the column containing the label
    labelColumn = 'price'

    # Assembling the vectors and outputting the training set
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol='features')
    output = assembler.transform(dataframe)
    vectorizedDF = output.select('features', col(labelColumn).alias('label'), 'window_num')
    
    # Gathering the total RMSE from all windows
    total_RMSE = []
    
   # Looping over windows, splitting into train/test sets, and training and evaluating a model on each set
    for window in range(1, num_windows+1):
        
        # Subsetting the dataframe into the window
        dataWindow = vectorizedDF.filter(col('window_num') == window).drop('window_num')

        # Splitting into train/testing sets
        trainWindow = sqlContext.createDataFrame(dataWindow.head(num_training_rows), dataWindow.schema)
        testWindow = dataWindow.subtract(trainWindow)
        
        # Fitting the model
        # Using L1 regularization for automatic feature selection
        lr = LinearRegression(elasticNetParam=1.0, regParam=0.03)
        model = lr.fit(trainWindow)
    
        # Gathering evaluation and summary metrics
        modelSummary = model.summary
        
        # Creating a plot of the predictions and actuals to see if there is a significant lag
        predictDF = model.transform(testWindow)  # Generating predictions
        total_RMSE.append(testRMSE)
        fig, ax = plt.subplots()
        ax.plot(predictDF.select('label').collect(), label='Label')
        ax.plot(predictDF.select('prediction').collect(), label='Prediction')
        plt.legend()
        plt.title('Test Set: Predictions and Actuals')
        
        # Reporting results
        print('Window', window)
        print('Training Size:', trainWindow.count())
        print('Testing Size:', testWindow.count())
        print("r2: %f" % modelSummary.r2)
        print("Training RMSE: %f" % modelSummary.rootMeanSquaredError)
        plt.show()  # Plot of actuals vs predictions
        print()
        
    print('Average RMSE for {0} windows: {1}'.format(num_windows, np.mean(total_RMSE)))
        
        
feature_columns = ['previous_hour_price', 'previous_hour_high_low_range', 'previous_hour_volume']
sliding_window_evaluation(dataframe=test, feature_columns=feature_columns, num_windows=3, test_size=0.2)

#################################################################################################################
##### Hyperparameter Tuning
# Grid Search - Spark ML
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Defining our parameter grid
paramGrid = (ParamGridBuilder()
  .addGrid(randomForest.numTrees, [10, 30, 100, 300])
  .addGrid(randomForest.maxDepth, [3, None])
  .build()
)

# Cross validation with the parameter grid
crossval = CrossValidator(estimator=randomForest,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)

# Reporting the number of nodes on the cluster
print('Number of nodes on the cluster:', sc._jsc.sc().getExecutorMemoryStatus().size())

# Performing the grid search
cvModel = crossval.fit(trainingDataset)

# Grabbing the best parameters
bestModelParams = cvModel.bestModel._java_obj.parent()

# Reporting the best obtained parameters
print('Hyperparameters for the best model:')
print('Number of Trees:', bestModelParams.getNumTrees())
print('Max Depth:', bestModelParams.getMaxDepth())


# Grid Search - Spark Scikit-Learn
import spark_sklearn
from sklearn.ensemble import RandomForestClassifier

# Ensuring our Spark context exists in the sc variable
# This is likely unnecessary in a Databricks cluster
sc = pyspark.SparkContext.getOrCreate()

randomForest = RandomForestClassifier()

# Defining our parameter grid
parameters = {'n_estimators': [10, 30, 100, 300],
              'max_depth': [3, None]
              'max_features': [1, 3, None],
              'min_samples_leaf': [1, 3, 10],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

# Cross validation with the parameter grid
model = spark_sklearn.GridSearchCV(sc, randomForest, parameters, refit=True, cv=3)

# Reporting the cluster size
print('Number of nodes on the cluster:', sc._jsc.sc().getExecutorMemoryStatus().size())

# Performing the grid search
model.fit(X, y)

# Reporting the parameters of the best model
model.best_estimator_

#################################################################################################################
##### Model Diagnostics
# Summary of a typical model
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(trainingSet)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


#################################################################################################################
##### Model Training
# Prediction Intervals with Quantile Regression in MML/LightGBM
def prediction_with_intervals(train, test, confidence_level=0.95):
    '''
    Trains LightGBM models and creates predictions on the data
  
    Input:
      - train: The training set with the label and vectorized features
      - test: The testing set with the label and vectorized features
      - confidence_level: The percent confidence level for the quantile regression
    
    Output: 
      - A Spark dataframe with the point estimates and upper/lower bounds
  
    TODO: Add more inputs for hyperparameter tuning
    '''
    from mmlspark import LightGBMRegressor
  
    # Calculating the upper/lower buffer for the quantile regressions
    alpha_buffer = (1 - confidence_level) / 2
  
    # Training all three models
    # Lower bound of 95% confidence interval
    model_lower_bound = LightGBMRegressor(application='quantile',
                                          alpha=confidence_level - alpha_buffer,
                                          learningRate=0.3).fit(train)

    # Upper bound of 95% confidence interval
    model_upper_bound = LightGBMRegressor(application='quantile',
                                          alpha=confidence_level + alpha_buffer,
                                          learningRate=0.3).fit(train)

    # Point prediction
    model = LightGBMRegressor(application='regression',
                             learningRate=0.3).fit(train)
    
    # Scoring on the testing set and assembling the results
    point_predictions = model.transform(test)
    upper_predictions = model_upper_bound.transform(test).withColumnRenamed('prediction', 'UpperBound')
    lower_predictions = model_lower_bound.transform(test).withColumnRenamed('prediction', 'LowerBound')
    
    # Assembling the results
    scored_data = (point_predictions.join(upper_predictions, ['label', 'features'])
                                    .join(lower_predictions, ['label', 'features']))
    
    return scored_data
