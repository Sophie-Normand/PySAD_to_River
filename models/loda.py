import anomaly
import numpy as np

from river import utils

class LODA(anomaly.base.AnomalyDetector):

    def __init__(self, num_bins=10, num_random_cuts=100):
        self.to_init = True
        self.n_bins = num_bins
        self.n_random_cuts = num_random_cuts
        
    def learn_one(self, x, y=None):
        """Fits the model to next instance.
        
        Parameters
        ----------
        X
            The instance to fit (np.float array of shape (num_features,)).
        y 
            Ignored since the model is unsupervised (Default=None).
            
        Returns
        -------
            object: Returns the self.
        """

        x_array = np.array(list(x.values()))
        
        if self.to_init:
            self.num_features = x_array.shape[0]
            self.weights = np.ones(self.n_random_cuts,dtype=np.float) / self.n_random_cuts
            self.projections_ = np.random.randn(self.n_random_cuts, self.num_features)
            self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

            n_nonzero_components = np.sqrt(self.num_features)
            self.n_zero_components = self.num_features - np.int(n_nonzero_components)

            self.to_init = False

        x_array = x_array.reshape(1,-1)
        for i in range(self.n_random_cuts):
            rands = np.random.permutation(self.num_features)[:self.n_zero_components]
            self.projections_[i, rands] = 0.
            projected_data = self.projections_[i, :].dot(x_array.T)
            self.histograms_[i, :], self.limits_[i, :] = np.histogram(projected_data, bins=self.n_bins, density=False)
            self.histograms_[i, :] += 1e-12
            self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

        return self
        
    def score_one(self, x):
        """Scores the anomalousness of the next instance.
        
        Parameters
        ----------
        X
            The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances (np.float array of shape (num_features,)).
            
        Returns
        -------
            float: The anomalousness score of the input instance.
        """
        
        x_array = np.array(list(x.values()))
        
        if self.to_init:
            self.num_features = x_array.shape[0]
            self.weights = np.ones(self.n_random_cuts,dtype=np.float) / self.n_random_cuts
            self.projections_ = np.random.randn(self.n_random_cuts, self.num_features)
            self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

            n_nonzero_components = np.sqrt(self.num_features)
            self.n_zero_components = self.num_features - np.int(n_nonzero_components)

            self.to_init = False

        x_array = x_array.reshape(1, -1)

        pred_scores = np.zeros([x_array.shape[0], 1])
        for i in range(self.n_random_cuts):
            projected_data = self.projections_[i, :].dot(x_array.T)
            inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, 0] += -self.weights[i] * np.log(self.histograms_[i, inds])
        pred_scores /= self.n_random_cuts

        return pred_scores.ravel()[0]
