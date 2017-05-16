import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plotter(object):
    '''A class that is initialised with a dataframe and can be used to plot histograms of all variables etc'''
    def __init__(self,df,outputDir,exceptions=[],binDict={}):

        self.df=df
        self.outputDir=outputDir

        self.binDict = binDict
        
        #In the case of exceptions drop them from the data frame
        self.exceptions = exceptions
        for drop in exceptions:
            self.df.drop(drop,axis=1,inplace=True)

    def plotAllHists1D(self,extraExceptions=[]):
        '''A function to plot sensible 1D histograms of all the columns of a dataframe excluding exceptions'''

        out = os.path.join(self.outputDir,'hists1d')
        if not os.path.exists(out): os.makedirs(out)

        for var in self.df.keys():
            if var in extraExceptions: continue
            print var
            self.df.hist(var)
            #plt.show()
            plt.savefig(os.path.join(out,var+'.pdf'))
            plt.close()

        return True

    def plotAllHists2D(self,varWrt,extraExceptions=[],binsVarWrt=(10,None,None)):
        '''A function to plot sensible 2D histograms of all the columns of a dataframe'''

        out = os.path.join(self.outputDir,'hists2d_'+varWrt)
        if not os.path.exists(out): os.makedirs(out)
        out2 = os.path.join(out,'hists1dweighted'+varWrt)
        if not os.path.exists(out2): os.makedirs(out2)


        for var in self.df.keys():
            if var in extraExceptions: continue
            #self.df.hist2d(varWrt,var)
            #Find the variable limits
            maxVar = round(max(self.df[var]),-1) + 10
            minVar = round(min(self.df[var]),-1)

            plt.hist2d(self.df[varWrt],self.df[var],bins=[binsVarWrt[0],20],range=[[binsVarWrt[1],binsVarWrt[2]],[minVar,maxVar]])
            #plt.show()
            plt.savefig(os.path.join(out,var+'.pdf'))
            plt.close()
            plt.hist(self.df[var],weights=self.df[varWrt])
            plt.savefig(os.path.join(out2,var+'.pdf'))
            plt.close()

        return True


    def correlations(self, subset=None, extraStr='',**kwds):
        """Calculate pairwise correlation between features.
        
        Extra arguments are passed on to DataFrame.corr()
        """
        # simply call df.corr() to get a table of
        # correlation values if you do not need
        # the fancy plotting
        # df = copy.deepcopy(data)
        #
        # for name in df.columns.values.tolist():
        #     #print name
        #     if name not in df.columns.values.tolist(): continue
        #     if 'score_' in name:
        #         df = df.drop(name,1)
        #     elif 'index' in name or 'Unnamed' in name:
        #         df = df.drop(name,1)

        # data.columns.values.tolist()
        #exit()
        
        if subset==None:
            corrmat = self.df.corr(**kwds)
        else:
            #ignore the things that aren't in the subset
            corrmat = self.df[subset].corr(**kwds)

        fig, ax1 = plt.subplots(ncols=1, figsize=(9,8))
        
        opts = {'cmap': plt.get_cmap("RdBu_r"),
                'vmin': -1, 'vmax': +1}
        heatmap1 = ax1.pcolor(corrmat, **opts)
        plt.colorbar(heatmap1, ax=ax1)

        ax1.set_title("Correlations")

        labels = corrmat.columns.values
        for ax in (ax1,):
            # shift location of ticks to center of the bins
            ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
            ax.set_yticklabels(labels, minor=False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputDir,'correlations'+extraStr+'.pdf'))
        plt.close()


