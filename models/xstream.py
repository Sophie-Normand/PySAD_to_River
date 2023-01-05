import anomaly
        
from utils.math import get_minmax_array
from utils.math import get_minmax_array_dico
#import math     #import math pour utiliser get_minmax_array

#from utils import get_minmax_array
from river.utils import dict2numpy
from river.utils import numpy2dict
import numpy as np
import math
import random


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
#        deltamax = np.ones(num_components) * 0.5
#        deltamax[np.abs(deltamax) <= 0.0001] = 1.0
        deltamax = {}
        for i in range (num_components):
            deltamax[i] = 0.5
#        deltamax = dict([[i, 0.5] for i in range (num_components)])
        deltamax = {feature: 1.0 if (value<=0.0001) else value for feature, value in deltamax.items()}
#        deltamax = [1/2] * num_components
#        deltamax_abs = [abs(x) for x in deltamax]
#        deltamax[deltamax_abs[0] <= 0.0001] = 1.0
        # ECRIRE DELTAMAX SOUS FORME DE DICO
        self.window_size = window_size
        self.hs_chains = _HSChains(
            deltamax=deltamax,
            n_chains=n_chains,
            depth=depth)

        self.step = 0
        self.cur_window = []
    #    self.cur_window = {}
        self.ref_window = None


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
        self.cur_window.append(X)

    #    self.ref_window[self.step] = X

        self.hs_chains.fit(X)

        if self.step % self.window_size == 0:
            self.ref_window = self.cur_window
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

    #    x_array = dict2numpy(X)
    #    print(x_array)
    #    score = self.hs_chains.score(X).flatten()
        score = self.hs_chains.score(X)
        #score = self.hs_chains.score(X)


        return score

    def _compute_deltamax(self):
        # mx = np.max(np.concatenate(self.ref_window, axis=0), axis=0)
        # mn = np.min(np.concatenate(self.ref_window, axis=0), axis=0)
    #    mn, mx = get_minmax_array(np.concatenate(self.ref_window, axis=0))
    #    deltamax = (mx - mn) / 2.0
    #    deltamax[abs(deltamax) <= 0.0001] = 1.0

        dico_min_max = {}
        for i in range(len(self.ref_window)):
            temp = {(i,feature):self.ref_window[i][feature] for feature in self.ref_window[i].keys()}
            dico_min_max = {**dico_min_max,**temp}
        mn, mx = get_minmax_array_dico(dico_min_max)
        deltamax = {key: (mx[key] - mn[key])/2.0 for key in mx.keys()}
        deltamax = {feature: 1.0 if (value<=0.0001) else value for feature, value in deltamax.items()}
        return deltamax


class _Chain:

    def __init__(self, deltamax, depth):
        
        k = len(deltamax)

        self.depth = depth
#        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.fs = [random.randint(0, k-1) for d in range(depth)]
        self.cmsketches = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth

        self.deltamax = deltamax  # feature ranges
#        self.rand_arr = np.random.rand(k)
#        self.rand_arr = random.random(k)
        self.shift ={}
        self.rand_arr = []
        for key in self.deltamax.keys():
            rnd = random.random()
            self.shift[key] = rnd * self.deltamax[key]
            self.rand_arr.append(rnd)
#        self.shift = self.rand_arr * deltamax

        self.is_first_window = True

    def fit(self, x):								#ajouter fit dans le AnomalyDetector
#        prebins = np.zeros(x.shape, dtype=np.float)
#        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        prebins ={}
        depthcount = {}
        for i in range (len(self.deltamax)):
            depthcount[i] = 0
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                for j in range(len(x)):
                    prebins[(j, f)] = (x[f] + self.shift[f]) / self.deltamax[f]
            else:
                for j in range(len(x)):
                    prebins[(j, f)] = 2.0 * prebins[(j, f)] - \
                    self.shift[f] / self.deltamax[f]

            if self.is_first_window:
                cmsketch = self.cmsketches[depth]
                
                key_0 = []
                key_1 = []
                for t in range(list(prebins.keys())[-1][0]+1):
                    key_0.append(t)
                for t in range(list(prebins.keys())[-1][1]+1):
                    key_1.append(t)
                for i in range(len(key_0)):
                    l_index = tuple(list(prebins.values())[i::len(key_1)])

                #for prebin in prebins:
#                    l_index = tuple(np.floor(prebin).astype(np.int))
                #    l_index = tuple(math.floor(prebin).astype(int))
                    if l_index not in cmsketch:
                        cmsketch[l_index] = 0
                    cmsketch[l_index] += 1

                self.cmsketches[depth] = cmsketch

                self.cmsketches_cur[depth] = cmsketch

            else:
                cmsketch = self.cmsketches_cur[depth]

#                for prebin in prebins:
#                    l_index = tuple(np.floor(prebin).astype(np.int))
#                    l_index = tuple(math.floor(prebin).astype(int))
                key_0 = []
                key_1 = []
                for t in range(list(prebins.keys())[-1][0]+1):
                    key_0.append(t)
                for t in range(list(prebins.keys())[-1][1]+1):
                    key_1.append(t)
                for i in range(len(key_0)):
                    l_index = tuple(list(prebins.values())[i::len(key_1)])
                    if l_index not in cmsketch:
                        cmsketch[l_index] = 0
                    cmsketch[l_index] += 1

                self.cmsketches_cur[depth] = cmsketch

        return self

    def bincount(self, x):
#        scores = np.zeros((x.shape[0], self.depth))
#        prebins = np.zeros(x.shape, dtype=np.float)
#        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        scores = {}
        prebins = {}
        depthcount = {}
        for i in range (len(self.deltamax)):
            depthcount[i] = 0

        for depth in range(self.depth):
            f = self.fs[depth]

            depthcount[f] += 1

            if depthcount[f] == 1:
                for j in range(len(x)):
                    prebins[(j, f)] = (x[f] + self.shift[f]) / self.deltamax[f]
            else:
                for j in range(len(x)):
                    prebins[(j, f)] = 2.0 * prebins[(j, f)] - \
                    self.shift[f] / self.deltamax[f]

            cmsketch = self.cmsketches[depth]

            for key in prebins:
                prebins[key] = math.floor(prebins[key])

            key_0 = []
            key_1 = []
            for t in range(list(prebins.keys())[-1][0]+1):
                key_0.append(t)
            for t in range(list(prebins.keys())[-1][1]+1):
                key_1.append(t)
            for i in range(len(key_0)):
                l_index = tuple(list(prebins.values())[i::len(key_1)])

    #        for i, prebin in enumerate(prebins):
    #            l_index = tuple(math.floor(prebin).astype(int))
                if l_index not in cmsketch:
                    scores[i, depth] = 0.0
                else:
                    scores[i, depth] = cmsketch[l_index]
        return scores

    def score(self, x):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(x)
    #    depths = np.array([d for d in range(1, self.depth + 1)])
        depths = {}
    #    for d in range (self.depth):
        for d in range (1, self.depth +1):
            depths[d-1] = d
        d_1 = dict(map(lambda x: (x[0], math.log2(1 + x[1])), scores.items()))
        for k in range (len(list(depths.keys()))):
            for j in range (list(d_1.keys())[-1][0]):
                d_1[j,k] += depths[k]
    #    scores = np.log2(1.0 + scores) + depths  # add 1 to avoid log(0)
        mini = {}
    #    print(list(d_1.keys())[-1][0])
        for j in range(list(d_1.keys())[-1][0]):
            mini[j] = - min([d_1[j,i] for i in range(list(d_1.keys())[-1][1]+1)])
    #    print(mini)
        return mini
    #    return -np.min(scores, axis=1)

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
    #    scores = np.zeros(x.shape[0])
        scores = []
        for ch in self.chains:
            scores += ch.score(x)
        
        scores = [elt/float(self.nchains) for elt in scores]

    #    scores /= float(self.nchains)
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
            for i in range(len(deltamax)):
                ch.shift = ch.rand_arr[i] * deltamax[i]