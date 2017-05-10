import os
import pandas as pd
import matplotlib.pyplot as plt

class Plotter(object):
    '''A class that is initialised with a dataframe and can be used to plot histograms of all variables etc'''
    def __init__(self,df,outputDir,exceptions=[],binDict={}):

        self.df=df
        self.outputDir=outputDir

        self.binDict = binDict
        self.exceptions = exceptions

    def plotAllHists1D(self,extraExceptions=[]):
        '''A function to plot sensible 1D histograms of all the columns of a dataframe excluding exceptions'''
        for var in self.df.keys():
            self.df.hist(var)
            #plt.show()
            plt.savefig(os.path.join(self.outputDir,var+'.pdf'))
            plt.close()
            print var
        pass

    def plotAllHists2D(self,var,extraExceptions=[]):
        '''A function to plot sensible 2D histograms of all the columns of a dataframe'''
        pass
