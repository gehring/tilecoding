import numpy as np

import theano
import theano.tensor as T

def sym_tiling_index(X,
                     input_index,
                     ntiles,
                     ntilings,
                     state_range,
                     offset = None,
                     hashing = None):
        s_range = [state_range[0][input_index].copy(), state_range[1][input_index].copy()]
        s_range[0] -= (s_range[1]-s_range[0])/(ntiles-1)

        if isinstance(ntiles, int):
            ntiles = np.array([ntiles]*len(input_index), dtype='uint')

        if offset == None:
            offset = np.empty((ntiles.shape[0], ntilings))
            for i in xrange(ntiles.shape[0]):
                offset[i,:] = -np.linspace(0, 1.0/ntiles[i], ntilings, False);
        if hashing == None:
            hashing = Theano_IdentityHash(ntiles)

        input_index = np.array(input_index, dtype='uint')
        size = ntilings*(hashing.memory)
        index_offset = (hashing.memory * np.arange(ntilings)).astype('int')


        nX = (X[:,input_index,:] - s_range[0][None,:,None])/(s_range[1]-s_range[0])[None,:,None]
        indices = T.cast(((offset[None,:,:] + nX)*ntiles[None,:,None]), 'int32')
        hashed_index = hashing.getHashedFunction(indices) + index_offset[None,:]
        return hashed_index, size
    


class Theano_IdentityHash(object):
    def __init__(self, dims):
        self.dims = dims
        self.memory = np.prod(dims)

    def getHashedFunction(self, indices):
        dims = np.cumprod(np.hstack(([1],self.dims[:0:-1]))).astype('int')[None,::-1,None]
        return T.sum(indices*dims, axis=1, keepdims = False)

class Theano_UNH(object):
    increment = 470
    def __init__(self, input_size, memory):
        self.rndseq = np.zeros(16384, dtype='int')
        self.input_size = input_size
        self.memory = memory
        for i in range(4):
            self.rndseq = self.rndseq << 8 | np.random.random_integers(np.iinfo('int16').min,
                                                                       np.iinfo('int16').max,
                                                                       16384) & 0xff

    def getHashedFunction(self, indices):
        rnd_seq = theano.shared(self.rndseq, borrow=False)
        a = self.increment*np.arange(self.input_size)
        index = indices + a[None,:,None]
        index = index - (T.cast(index, 'int64')/self.rndseq.size)*self.rndseq.size
        hashed_index = T.cast(T.sum(rnd_seq[index], axis=1, keepdims = False), 'int64')
        return hashed_index - (hashed_index/(int(self.memory)))*int(self.memory)


class Theano_Tiling(object):
    def __init__(self,
                 input_indicies,
                 ntiles,
                 ntilings,
                 hashing,
                 state_range,
                 bias_term = True):
        if hashing == None:
            hashing = [None]*len(ntilings)
        X = T.TensorType(dtype = theano.config.floatX, broadcastable = (False, False, True))('X')
        X.tag.test_value = np.random.rand(1, 2,1).astype('float32')
        tilings, sizes = zip(*[sym_tiling_index(X, in_index, nt, t, state_range, hashing = h,)
                   for in_index, nt, t, h in zip(input_indicies, ntiles, ntilings, hashing)])

        self.__size = int(sum(sizes))
        index_offset = np.zeros(len(ntilings), dtype = 'int')
        index_offset[1:] = np.cumsum(sizes)
        index_offset = np.hstack( [np.array([off]*t, dtype='int')
                                            for off, t in zip(index_offset, ntilings)])

        all_indices = T.cast(T.concatenate(tilings, axis=1), 'int32') + index_offset.astype('int')
        if bias_term:
            all_indices = T.cast(T.concatenate((all_indices, self.__size*T.ones((all_indices.shape[0], 1))), axis=1), 'int32')
            self.__size += 1

        self.proj = theano.function([X], all_indices, allow_input_downcast=True)

    def __call__(self, state):
        if state.ndim == 1:
#             state = state.reshape((1,-1,1))
            phi = self.proj(state[None,:,None])[0,:]
        else:
            phi = self.proj(state[:,:,None])
        return phi

    @property
    def size(self):
        return self.__size
