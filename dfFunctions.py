import pandas as pd

def expandArrays(df):

    for var in df.keys():
        if df[var].dtype.kind is 'O':

            #Assuming it's a row, so expand it out
            new= df[var].apply(pd.Series)
            new.columns = [var+str(n) for n in new.keys()]
            #drop the old one
            df = df.drop([var],axis=1)
            df = pd.concat([df,new],axis=1)

    return df

