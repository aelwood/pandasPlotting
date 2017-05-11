def classifyScore(score,classification=0):
    '''Returns 0 or 1 depending whether the score is above or below classification'''
    if score > classification: return 1
    else: return 0

def featureImportance(df,classifier,output):
    pass
