import sys

print('Python:', sys.version)

# Train/test split
# We'll hold out 80% for training and leave 20% for testing 
seed = 46
(trainingDF, testDF) = df.randomSplit([8.0, 2.0], seed)  # Normalizes if it doesn't sum up to 1


# Evaluating regression
# https://spark.apache.org/docs/2.2.0/ml-tuning.html
from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator().setLabelCol('label')
predictDF = model.transform(testDF)
testRMSE = regEval.evaluate(predictDF)
print('The model had a RMSE on the test set of {0}'.format(testRMSE))


# Grid Search CV
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