
from numpy.linalg import eig, eigh
import numpy as np
from numpy.random import rand, randint, randn
from scipy.signal import convolve2d
from sklearn.decomposition import TruncatedSVD
import time


import tensorflow as tf
class layer():
    rho = 0
    num_im = 0  # to help w/ normalization 
    def __init__(self,data, n='auto', k= 3, nf='all', stride = 1, out=True, **kw):
        self.k = k
        self.stride = stride
        print "making density matrix..."
        #self.get_rho(data)
        self.get_rho_fast(data)
        print "density matrix done!"
        self.in_channels = (data[0].shape[0] if len(data[0].shape) ==3 else 1)
        nmax = self.in_channels*self.k**2
        if (nf == 'all') or (nf >= nmax):
            self.num_filters = self.rho.shape[0]
            self.get_eigs0()
        else:
            self.num_filters = min(nf, nmax-1) 
            self.get_eigs()
        
        self.setups()
        
    def setups(self):
        self.output =0 
        self.setup_output()
        self.setup_normalized_output()
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def setup_output(self):
        self.ims = tf.placeholder(np.float32)
        self.flt = tf.placeholder(np.float32)
        self.output = tf.nn.conv2d(self.ims, self.flt, 
                        strides= [1,1,1,1],padding= "VALID")
        
    def setup_normalized_output(self):
        #x = tf.placeholder(dtype=tf.float32)
        mn, var = tf.nn.moments(self.ims,axes=[2,3],keep_dims=True)
        self.normalized_output = (self.ims-mn)/tf.sqrt(var)
        
    def get_output(self,ims,idx = 'all', batch = 5000):
        if idx == 'all':
            fls = self.filters
        else:
            fls = self.filters[idx]

        s1 = self.filters.shape
        k = self.k
        n = np.prod(s1)/s1[0]/k**2
        # to shape [out_chan, in_chan, x,y]
        f1 = fls.reshape(s1[0],n, k, k)
        # transpose to [x,y,in, out]
        f1 = f1.transpose((2,3,1,0)).astype(np.float32)
        out = []
        for i in range(0,len(ims), batch):
            im = np.array(ims[i:batch+i],dtype= np.float32).transpose((0,2,3,1)) 
            out += [self.output.eval({self.flt: f1, self.ims: im}).transpose((0,3,1,2))]
        return np.concatenate(out)
    
    def get_normalized_output(self,ims):
        return self.normalized_output.eval({self.ims: ims})
        
    def inc_rho(self, b):
        n = self.num_im
        self.rho = (n * self.rho + b)/(n+1.)
        self.num_im += 1
    
    def sub_rho(self,im):
        """NOTE: This function uses a fixed grid on the image to calculate rho 
        i.e. the stride for sub ims is k, same as sub im size. 
        To get fully accurate rho, stride 1 should be used, but it really won't matter in real images."""
        k = self.k
        s = np.array(im.shape)
        ch = s[0] # channels
        sx,sy = s[1:]/k
        # reshape to get sub ims
        p1 = im[:,:k*sx,:k*sy].reshape((ch,sx,k,sy,k)).transpose((1,3,0,2,4)) # now p[i,j] are sub-ims
        p1 = p1.reshape((sx*sy,ch*k*k))
        # ! memory intensive, but fast...
        #rho = (p1[:,np.newaxis]*p1[:,:,np.newaxis]).mean(0)
        rho = 0
        for p in p1:
            rho = rho + p*p[:,np.newaxis]
        return rho/p1.shape[0]
    
    def get_rho_fast (self, data):
        print "rho: processing images:",
        for im in data:
            #print im.shape
            for ki in range(0,self.k-1,self.stride):
                for kj in range(0,self.k-1,self.stride):
                    self.inc_rho( self.sub_rho(im[:,ki:,kj:]))
            
    def get_eigs0(self):
        self.eigs = eig(self.rho)
        # sort eigs from  largest to smallest  
        #idx = argsort(real(self.eigs[0]))[::-1]
        self.energies = -real(log(self.eigs[0]/self.eigs[0].sum()))
        idx = argsort(self.energies)
        self.energies = self.energies[idx]
        ### NOTE! filters are rows of self.filters, not columns like in eigs!!
        self.filters = self.eigs[1][:,idx].T  
        
    def get_eigs(self):
        self.eigs = TruncatedSVD(n_components=self.num_filters, n_iter=7, random_state=42)
        self.eigs.fit(self.rho) # 
        self.energies = -real(log(self.eigs.explained_variance_/self.eigs.explained_variance_.sum()))
        idx = argsort(self.energies)
        self.energies = self.energies[idx]
        ### NOTE! filters are rows of self.filters, not columns like in eigs!!
        self.filters = self.eigs.components_[idx]
        
    def truncate(self, nf):
        self.energies = self.energies[:nf]
        self.filters = self.filters[:nf]
        #
    def update_layer(self,data):
        self.get_rho_fast(data)
        self.get_eigs()
        
    
    def get_filter_outputs(self,im,idx = 'all'):
        """
        apply filters to an image `im`. If idx = [ni,... nf] given, only output of filters[idx] is returned
        Output: ndarray (images x filters)  
        """
        if idx =='all':
            eigvec = self.filters
        else: eigvec = self.filters[idx]
        #print "eig:", eigvec.shape, self.k, im.shape
        if len(im.shape) == 2:
            return np.array([convolve2d(im,i.reshape(self.k,self.k), mode='valid') for i in eigvec])
        else:
            nf0 = im.shape[0]
            k2 = self.k**2
            out = []
            for i in eigvec:
                # each row is the output of one filter from previous layer
                # It should be convolved with the corresponding rows in each eigvec
                out += [np.array([convolve2d(imf,i[ii*k2:(ii+1)*k2].reshape(self.k,self.k),
                                         mode='valid') for imf, ii in zip(im, range(nf0))]).sum(0)]
                # to sum over nf0 channels

            return np.array(out) # shape (nf1, x-k, y-k)
    
    def get_output_old(self,images):
        ii = 0
        out = []
        for im in images:
            #print ii,
            out += [self.get_filter_outputs(im)]
            ii+=1 
        return out

        
    ### Vizualization
    def viz_filters(self, n='all'):
        k = self.k
        if n=='all':
            F,E = self.filters, self.energies
        else:
            F,E = self.filters[n], self.energies[n] 
        for c in range(self.in_channels):
            figure(figsize=(8,8))
            ii = 1
            print "For Input filter %d" %c
            k1 = int(sqrt(len(F)))+1
            for i,en in zip(F, E):
                subplot(k1,k1,ii)
                title('%.3g'%en)
                ii+=1
                imshow(real(i[c*k**2:(c+1)*k**2].reshape(k,k)),cmap='binary')
                xticks([])
                yticks([])
            show()
            
    
            

class DMN():
    def __init__(self,ims):
        """Create a Density Matrix Network ;)
        It will initialize with no layers. 
        you generate layers by invoking the self.create_layer() method, 
        creating an instance of the `layer` class.  
        ims: set of input images
        """
        # print "Adding random noise to avoid high degeneracies..."
        if len(ims[0].shape)==3:
            self.output = [(i+rand(*i.shape)).transpose((2,0,1)) for i in ims]
        else:
            self.output = [(i+rand(*i.shape))[np.newaxis,:] for i in ims]
        self.layers = []
        self.__setup_pooling()
        
    def __setup_pooling(self, k = 2):
        self.out_ = tf.placeholder(dtype=tf.float32)
        self.pool_k_ = tf.placeholder(dtype=tf.int8)
        self._max_pool = tf.nn.max_pool(self.out_, 
                                     ksize=[1,k,k,1], 
                                     strides= [1,k,k,1],
                                     padding= "SAME")#, data_format="NCHW")
        self._avg_pool = tf.nn.avg_pool(self.out_, 
                                     ksize=[1,k,k,1], 
                                     strides= [1,k,k,1],
                                     padding= "SAME")#, data_format="NCHW")
    def pooling(self, data, type = 'max'):
        assert net.output.dtype == np.float32
        if type == 'max':
            return self._max_pool.eval({self.out_: data.transpose(0,2,3,1) }).transpose(0,3,1,2)
        elif type == 'avg':
            return self._avg_pool.eval({self.out_: data.transpose(0,2,3,1) }).transpose(0,3,1,2)
        
    def create_layer(self,**kw):
        print "Propagating through last layer"
        if len(self.layers)>0:
            pass # self.output = self.layers[-1].get_output(self.output)
        self.layers +=[layer(self.output, **kw)]
        
    def get_filter(self,l1, l2,n1, n2):
        ei_sq = l2.filters[n2].reshape((len(l1.filters),l2.k, l2.k))
        return convolve2d(ei_sq[0], l1.filters[n1].reshape((l1.k,l1.k)))
    
    def get_mask(self,a, nlay):
        l = self.layers[nlay-1]
        n = len(l.filters)
        c = l.filters.shape[1]/(l.k**2)
        if len(a.shape)==1:
            k = self.layers[nlay].k
            a = a.reshape((n,k, k))
        if nlay ==1: # second layer
            sh = (c,l.k,l.k)
            
        else:
            sh = (self.layers[nlay-2].filters.shape[0],l.k,l.k)
        
        # each filter in nlay-1 connects to nlay-2 filters, 
        # conv2d must yield #nlay-2 outputs 
        out =[0]*sh[0] 
        for i in range(n):
            lf = l.filters[i].reshape(sh)
            for j in range(sh[0]):
                out[j] += convolve2d(a[i], lf[j])
                
        return np.array(out)
    
    def image_filter(self, nlay, nfil):
        """recursively go down the network and generate the mask on image. 
        !!! Needs to account for pooling layers"""
        out = self.layers[nlay].filters[nfil]
        for n in range(nlay)[::-1]:
            #print n,
            out = self.get_mask(out, n+1)
        return out
        
    def max_pooling(self, size = 2):
        # 1) get output of previous layer
        # 2) downsample using max   
        out = []
        for fim in self.output: 
            filter_out = [] # for each filter, we maxpool the output
            for im in fim:
                s = im.shape
                ds = np.array([[im[i*size:(i+1)*size,j*size:(j+1)*size].max() \
                            for j in range(s[1]/size)] for i in range(s[0]/size)])
                filter_out+=[ds]
            out +=[np.array(filter_out)]
        self.output = np.array(out)


def get_mask(self,a, nlay):
        l = self.layers[nlay-1]
        n = len(l.filters)
        c = l.filters.shape[1]/(l.k**2)
        if len(a.shape)==1:
            k = self.layers[nlay].k
            a = a.reshape((n,k, k))
        if nlay ==1: # second layer
            sh = (c,l.k,l.k)
            
        else:
            sh = (self.layers[nlay-2].filters.shape[0],l.k,l.k)
        
        # each filter in nlay-1 connects to nlay-2 filters, 
        # conv2d must yield #nlay-2 outputs 
        out =[0]*sh[0] 
        for i in range(n):
            lf = l.filters[i].reshape(sh)
            for j in range(sh[0]):
                out[j] += convolve2d(a[i], lf[j])
                
        return np.array(out)
    
class tictoc():
    prev = 0
    current = 0
    def tic(self):
        self.current = time.time()
        
    def toc(self):
        self.prev = self.current + 0.
        self.current = time.time()
        print "time:", self.current-self.prev
        
tc= tictoc()

def get_ims(fnam = '/home/ivplroot/Downloads/DeepLearning/dataset/mnist_test.csv'):
    f = open(fnam, 'r')
    a = [np.int0(i.split(',')) for i in f.readlines()]
    f.close()
    label = [i[0] for i in a]
    im_arr = [i[1:].reshape((28,28)) for i in a] # 
    return np.array(label), im_arr

