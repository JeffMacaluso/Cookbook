import sys
import time

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('Python:', sys.version)

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
from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator().setLabelCol('label')
crossval = CrossValidator(estimator=model,
                          evaluator=regEval,
                          numFolds=3)

# Re-fitting on the entire training set
cvModel = crossval.fit(trainingDF)

# Sliding window for time series model evaluation
def sliding_test(dataframe, feature_columns, num_windows=5, test_size=0.2):
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
        
        print('Window', window)
        print('Training Size:', trainWindow.count())
        print('Testing Size:', testWindow.count())
        print("r2: %f" % modelSummary.r2)
        print("Training RMSE: %f" % modelSummary.rootMeanSquaredError)
        print()
        
        
feature_columns = ['previous_hour_price', 'previous_hour_high_low_range', 'previous_hour_volume']
sliding_test(dataframe=test, feature_columns=feature_columns, num_windows=3, test_size=0.2)

#################################################################################################################
##### Hyperparameter Tuning
# Grid Search
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Defining our parameter grid
paramGrid = (ParamGridBuilder()
  .addGrid(model.params, [1, 5, 15])
  .build()
)

# Cross validation with the parameter grid
crossval = CrossValidator(estimator=model,
                          estimatorParamMaps=paramGrid,
                          evaluator=regEval,
                          numFolds=3)

# Re-fitting on the entire training set
cvModel = crossval.fit(trainingDF)

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
