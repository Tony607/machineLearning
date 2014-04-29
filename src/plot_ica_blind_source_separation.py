"""
=====================================
Blind source separation using FastICA
=====================================

An example of estimating sources from noisy data.

:ref:`ICA` is used to estimate sources given noisy measurements.
Imagine 2 instruments playing simultaneously and 2 microphones
recording the mixed signals. ICA is used to recover the sources
ie. what is played by each instrument.

"""
print(__doc__)

import numpy as np
import pylab as pl
from sklearn.decomposition import FastICA

###############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 10, n_samples)
s1 = 0.53*np.cos(1.73 * time)  # Signal 1 : sinusoidal signal
s2 = 1.*np.sign(np.sin(1.22 * time+0.5))  # Signal 2 : square signal
s3 = np.sin(np.sin(2.92 * time+7.1))  # Signal 2 : square signal
S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1., 3.5, 1.], [0.5, 2., 1.], [1.5, 1., 1.]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations
# Compute ICA
ica = FastICA()
S_ = ica.fit(X).transform(X)  # Get the estimated sources
A_ = ica.mixing_  # Get estimated mixing matrix
print A_
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

###############################################################################
# Plot results
pl.figure()
pl.subplot(3, 1, 1)
pl.plot(S)
pl.title('True Sources')
pl.subplot(3, 1, 2)
pl.plot(X)
pl.title('Observations (mixed signal)')
pl.subplot(3, 1, 3)
pl.plot(S_)
pl.title('ICA estimated sources')
pl.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
pl.show()