import sys
import time

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('Python:', sys.version)

#################################################################################################################
##### Cross Validation
# Train/test split
seed = 46
(trainingDF, testDF) = df.randomSplit([8.0, 2.0], seed)  # 80/20 split

# Evaluating regression
# https://spark.apache.org/docs/2.2.0/ml-tuning.html
from pyspark.ml.evaluation import RegressionEvaluator

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