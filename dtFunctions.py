from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
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

def decisionTree(df,classifier,output,subset=None,drawTree=False,**kwargs):

    out = os.path.join(output,'decisionTree')
    if not os.path.exists(out): os.makedirs(out)

    dt =  DecisionTreeClassifier(**kwargs)

    y = df[classifier]#[:-1000]
    if subset!=None:
        X = df[subset]#[:-1000]
    else:
        X = df.drop(classifier,axis=1)

    dt.fit(X,y)

    if drawTree:
        #Print out the tree
        outTree=os.path.join(out,'tree.dot')
        export_graphviz(dt,out_file=outTree,feature_names=X.columns.values.tolist(),\
                        filled=True, rounded=True,special_characters=True)
        os.system('dot -Tpng '+outTree+' -o '+os.path.join(out,'tree.png'))

    #predicted = dt.predict(X)
    predicted = dt.predict(X)#[-1000:])
    #df=df[-1000:]

    print 'Pre tree profit:',df['adjustedprofit'].sum()
    print 'Post tree 1 profit:',df[predicted==1]['adjustedprofit'].sum()
    print 'Post tree 0 profit:',df[predicted==0]['adjustedprofit'].sum()
    print 'Godlike 1 profit:',df[df['y_adjustedprofit']==1]['adjustedprofit'].sum()
    print 'Godlike 0 profit:',df[df['y_adjustedprofit']==0]['adjustedprofit'].sum()
    exit()


    pass

