import sys
import time

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('Python:', sys.version)

#################################################################################################################
##### Data Preparation
# Vectorizing a training set before feeding into a ML model
from pyspark.ml.feature import VectorAssembler

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
