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
##### Evaluation Plots

# Residuals
def plot_residuals(model, values, labels):
    '''
    Creates two plots: Actual vs. Predicted and Residuals
    '''
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
    '''
    Plots the learning curve of a model using 3-fold Cross Validation
    '''
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
    '''
    Plots the validation curve of a model using 3-fold Cross Validation
    '''
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


# Ensemble Model's Feature Importance
def plot_ensemble_feature_importance(model, features, top_n_features=None, plot_size=(15, 15), return_dataframe=False):
    '''
    Plots the scaled feature importance for an ensemble model. Returns a data frame of these if requested.
    
    Inputs:
        - model (): Scikit-learn, LightGBM, or xgboost ensemble model
        - features (dataframe): The training features
        - top_n_features (int): The number of features to plot
        - plot_size (tuple): The size of the output plot
        - return_dataframe (bool): Whether or not to output a data frame in the results
    '''
    # Putting the feature importances into a data frame
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=features.columns,
                                       columns=['Feature Importance']).sort_values('Feature Importance',
                                                                                   ascending=False)
    # Calculating the scaled importances and adding it as a column
    scaled_feature_importances = 100.0 * (feature_importances['Feature Importance'] / feature_importances['Feature Importance'].max())
    feature_importances['Scaled Feature Importance'] = scaled_feature_importances
    
    # Plotting the feature importances
    plt.figure(figsize=plot_size)
    if top_n_features != None:
        feature_importances['Scaled Feature Importance'][:top_n_features].sort_values(ascending=True).plot.barh()
    else:
        feature_importances['Scaled Feature Importance'].sort_values(ascending=True).plot.barh()
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
    # Returning the feature importance data frame if requested
    if return_dataframe == True:
        return feature_importances


# Visualizing the decision tree
def plot_decision_tree(model, feature_names=None):
    '''
    Plots the decision tree from a scikit-learn DecisionTreeClassifier or DecisionTreeRegressor
    Requires graphviz: https://www.graphviz.org
    
    Notes on decision tree visualization:
        - The Gini score is the level of "impurity" of the node. 
            - Scores closer to 0.5 are more mixed, whereas scores closer to 0 are more homogenous
        - For classification, the colors correspond to different classes
            - The shades are determined by the Gini score. Nodes closer to 0.5 will be lighter.
        - Values contain the number of samples in each category
    
    TODO: Add notes on decision tree visualization for regression
    '''
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO()
    
    export_graphviz(model, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names=feature_names)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    display(Image(graph.create_png()))
    
    
# Visualizing feature importance with SHAP (SHapely Additive exPlanations)
def explain_features_shap(model, features, max_features_to_show=15):
    '''
    TODO: Write explanation
    '''
    import shap
    
    # Explaining the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Summarizing the effect of all features
    print('Overall Feature Importance')
    shap.summary_plot(shap_values, features, plot_type="bar")  # Bar plot of feature importance
    
    print('-----------------------------------------------------------------')
    print('Detailed Feature Importance')
    shap.summary_plot(shap_values, features)  # Structured scatter plot
    
    # Showing the effect of each feature across the whole dataset if there are not too many features
    if features.shape[1] <= max_features_to_show:
        print('-----------------------------------------------------------------')
        print('SHAP values for individual features')
        for name in features.columns:
            shap.dependence_plot(name, shap_values, features, display_features=features)

            
# Confusion matrix
def plot_confusion_matrix(label, predictions, classes, normalize=False):
    '''
    TODO: Write docstring
    '''
    import itertools
    from sklearn import metrics
    
    # Calculating the confusion matrix
    cm = metrics.confusion_matrix(label, predictions)
    
    # Normalizing if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix')

    print(cm)

    # Plotting the confusion matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Adding the text properties to the graph
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black',
        )

    # Additional graph properties
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
