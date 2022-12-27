import anomaly
        
from utils.math import get_minmax_array
from utils.math import get_minmax_array_dico
#import math     #import math pour utiliser get_minmax_array

#from utils import get_minmax_array
from river.utils import dict2numpy
from river.utils import numpy2dict
import numpy as np


from transform.projection.streamhash_projector import StreamhashProjector
#from pysad.utils import get_minmax_array


class xStream(anomaly.base.AnomalyDetector):
    """The xStream model for row-streaming data :cite:`xstream`. It first projects the data via streamhash projection. It then fits half space chains by reference windowing. It scores the instances using the window fitted to the reference window.
    
    Parameters
    ----------
    n_components
        Number of components for streamhash projection (Default=100).
    n_chains
        Number of half-space chains (Default=100).
    depth
        Maximum depth for the chains (Default=25).
    window_size 
        Size (and the sliding length) of the reference window (Default=25).
    
    """

    def __init__(
            self,
            num_components=100,
            n_chains=100,
            depth=25,
            window_size=25):
        self.streamhash = StreamhashProjector(num_components=num_components)
        deltamax = np.ones(num_components) * 0.5
        deltamax[abs(deltamax) <= 0.0001] = 1.0
        self.window_size = window_size
        self.hs_chains = _HSChains(
            deltamax=deltamax,
            n_chains=n_chains,
            depth=depth)

        self.step = 0
        self.cur_window = []
    #    self.cur_window = {}
    #    self.ref_window = None
        self.ref_window = {}


    def merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res

    def learn_one(self, x, y=None):
        """Fits the model to next instance.
        
        Parameters
        ----------
        X
            Instance to learn.
        y
            Ignored since the model is unsupervised (Default=None).
        
        """
        self.step += 1

        #X = self.streamhash.learn_transform_one(X)
    #    x_array = dict2numpy(x)
    #    X = numpy2dict(x_array)
        X = self.streamhash.learn_one(x)
        X = self.streamhash.transform_one(X)

        #X = X.reshape(1, -1)
    #    x_array = x_array.reshape(1,-1)
    #    self.cur_window.append(X)
        self.cur_window.append(X.values)

    #    self.ref_window[self.step] = X

        self.hs_chains.fit(X)

        if self.step % self.window_size == 0:
        #    self.ref_window = self.cur_window
            self.ref_window[self.step] = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.hs_chains.set_deltamax(deltamax)
            self.hs_chains.next_window()

        return self

    def score_one(self, x):
        """Scores the anomalousness of the next instance.
        
        Parameters
        ----------
        X 
            Instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.
        
        """
    #    x_array = dict2numpy(x)
    #    X = numpy2dict(x_array)
        X = self.streamhash.learn_one(x)
        X = self.streamhash.transform_one(X)
    #    print(X)
        #X = X.reshape(1, -1)
        #x_array = x_array.reshape(1,-1)

        x_array = dict2numpy(X)
    #    print(x_array)
        score = self.hs_chains.score(x_array).flatten()
        #score = self.hs_chains.score(X)


        return score

    def _compute_deltamax(self):
        # mx = np.max(np.concatenate(self.ref_window, axis=0), axis=0)
        # mn = np.min(np.concatenate(self.ref_window, axis=0), axis=0)
    #    mn, mx = get_minmax_array(np.concatenate(self.ref_window, axis=0))
        mn, mx = get_minmax_array_dico(self.ref_window)

        deltamax = (mx - mn) / 2.0
        deltamax[abs(deltamax) <= 0.0001] = 1.0

        print(deltamax)

        return deltamax


class _Chain:

    def __init__(self, deltamax, depth):
        
        k = len(deltamax)

        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth

        self.deltamax = deltamax  # feature ranges
        self.rand_arr = np.random.rand(k)
        self.shift = self.rand_arr * deltamax

        self.is_first_window = True

    def fit(self, x):								#ajouter fit dans le AnomalyDetector
        prebins = np.zeros(x.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:, f] = (x[:, f] + self.shift[f]) / self.deltamax[f]
            else:
                prebins[:, f] = 2.0 * prebins[:, f] - \
                    self.shift[f] / self.deltamax[f]

            if self.is_first_window:
                cmsketch = self.cmsketches[depth]
                for prebin in prebins:
                    l_index = tuple(np.floor(prebin).astype(np.int))
                    if l_index not in cmsketch:
                        cmsketch[l_index] = 0
                    cmsketch[l_index] += 1

                self.cmsketches[depth] = cmsketch

                self.cmsketches_cur[depth] = cmsketch

            else:
                cmsketch = self.cmsketches_cur[depth]

                for prebin in prebins:
                    l_index = tuple(np.floor(prebin).astype(np.int))
                    if l_index not in cmsketch:
                        cmsketch[l_index] = 0
                    cmsketch[l_index] += 1

                self.cmsketches_cur[depth] = cmsketch

        return self

    def bincount(self, x):
        scores = np.zeros((x.shape[0], self.depth))
        prebins = np.zeros(x.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:, f] = (x[:, f] + self.shift[f]) / self.deltamax[f]
            else:
                prebins[:, f] = 2.0 * prebins[:, f] - \
                    self.shift[f] / self.deltamax[f]

            cmsketch = self.cmsketches[depth]
            for i, prebin in enumerate(prebins):
                l_index = tuple(np.floor(prebin).astype(np.int))
                if l_index not in cmsketch:
                    scores[i, depth] = 0.0
                else:
                    scores[i, depth] = cmsketch[l_index]

        return scores

    def score(self, x):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(x)
        depths = np.array([d for d in range(1, self.depth + 1)])
        scores = np.log2(1.0 + scores) + depths  # add 1 to avoid log(0)
        return -np.min(scores, axis=1)

    def next_window(self):
        self.is_first_window = False
        self.cmsketches = self.cmsketches_cur
        self.cmsketches_cur = [{} for _ in range(self.depth)] * self.depth


class _HSChains:
    def __init__(self, deltamax, n_chains=100, depth=25):
        self.nchains = n_chains
        self.depth = depth
        self.chains = []

        for i in range(self.nchains):

            c = _Chain(deltamax=deltamax, depth=self.depth)
            self.chains.append(c)

    def score(self, x):
        scores = np.zeros(x.shape[0])
        for ch in self.chains:
            scores += ch.score(x)

        scores /= float(self.nchains)
        return scores

    def fit(self, x):
        for ch in self.chains:
            ch.fit(x)

    def next_window(self):
        for ch in self.chains:
            ch.next_window()

    def set_deltamax(self, deltamax):
        for ch in self.chains:
            ch.deltamax = deltamax
            ch.shift = ch.rand_arr * deltamax