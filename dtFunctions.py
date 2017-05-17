from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
import os

def classifyScore(score,classification=0):
    '''Returns 0 or 1 depending whether the score is above or below classification'''
    if score > classification: return 1
    else: return 0

def featureImportance(df,classifier,output,exceptions=[],error=False):
    '''Classify the variable importance for a given set of trees code is taken
    from:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html\
            #sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    '''

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=1000,
                                  random_state=0)

    # Make the array of features, X, and the classifier, y

    y = df[classifier]
    X = df.drop(classifier,axis=1)
    for e in exceptions:
        X.drop(e,axis=1,inplace=True)        

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    outTxtFile = open(os.path.join(output,'featureList.txt'),'w')
    outTxtFile.write("Feature ranking:\n")

    for f in range(X.shape[1]):
        outTxtFile.write("%d. feature %s (%f)" % (f + 1, X.columns.values.tolist()[indices[f]], importances[indices[f]])+'\n')

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    if error:
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
    else:
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig(os.path.join(output,'featureImportances.pdf'))
    plt.close()

    pass

