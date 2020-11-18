import numpy as np
from scipy.stats import poisson
from scipy.special import gammaln
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

# 2020-10-19 David Hoogerheide

def resample_poisson_dave(countarray, usecdf=True):
    # countarray is an N-D array of count values that will be resampled using Poisson sampling

    # save original shape for reshaping
    orgshape = countarray.shape

    # turn count array into a 1-D flat array (may not be necessary any more)
    allcts = np.ravel(countarray)

   
    # if we don't want uncertainty on the means taken into account:
    if usecdf:
        # Define empty resampled array
        newcts = np.empty_like(allcts)

        # find unique values of counts, and number of occurrences.
        # This allows only one CDF calculation for each count value
        cts, ncts = np.unique(allcts, return_counts=True)
        
        # define precision of CDF calculation
        npts = 100
        minvals = np.max((1e-6*np.ones_like(cts), cts -5*np.sqrt(cts)), axis=0)
        maxvals = np.max((10.0*np.ones_like(cts), cts + 5* np.sqrt(cts)), axis=0)
        mux = np.linspace(
                minvals,
                maxvals,
                npts + 1)
        
        # calculate log PDF for each unique count value
        logpdf_mu = cts*np.log(mux) - mux - gammaln(cts+1)

        # integrate to find CDF for each unique count value
        cdf_mu = cumtrapz(np.exp(logpdf_mu), mux, axis=0)

        # step through unique count values to do interpolation
        # TODO: this could be a lot faster if it can be done with matrix math
        for i, nct in enumerate(ncts):
            # interpolate random numbers in [0, 1) to the CDF to find the predicted
            # (random) rates mu (or lambda typically in a Poisson distribution)
            mu = np.interp(np.random.random(nct), cdf_mu[:,i], mux[:-1,i])

            # sample from the distributions 20201022: probably not right
            newct = poisson.rvs(mu)

            # insert resampled values into the resampled array
            newcts[allcts==cts[i]] = newct
    else:
        newcts = poisson.rvs(allcts)


    # Reshape and return
    return newcts.reshape(orgshape)
    
def resample_poisson(countarray, usecdf=False):
    eps = 1e-14 # just needs to be nonzero and small. checked with 1e-5 to 1e-300
    return np.random.negative_binomial(countarray + 1e-14, 0.5)