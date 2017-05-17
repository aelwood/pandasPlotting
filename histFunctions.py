import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def hist1dError(data,weights=[],**kwargs):
    #if there aren't any weights just set them to one
    if len(weights)==0: weights = np.ones(len(data))
    
    #n,bins = np.histogram(data, **kwargs)
    hist,bins = np.histogram(data, weights=weights, **kwargs)
    err2,bins = np.histogram(data, weights=weights*weights, **kwargs)
    err=np.sqrt(err2)

    plt.bar((bins[1:]+bins[:-1])/2,hist,yerr=err,width=bins[1]-bins[0],color='r',alpha=0.8)
    #plt.errorbar((bins[1:]+bins[:-1])/2,hist,yerr=err,drawstyle='steps-mid-')
    pass

