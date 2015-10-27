"""
Convolutional neural net on MNIST, modeled on 'LeNet-5',
http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.signal
from autograd import grad
from autograd.util import quick_grad_check
from six.moves import range
import gmm_util as gmm_util

convolve = autograd.scipy.signal.convolve

class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def make_nn_funs(input_shape, layer_specs, L2_reg):
    """ constructs network and returns loss function  """
    parser = WeightsParser()
    cur_shape = input_shape
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        parser.add_weights(layer, (N_weights,))

    def predict_distribution(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)
            cur_units   = layer.forward_pass(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)

        # compute distribution for each input
        params    = predict_distribution(W_vect, X)
        means     = params[:, :8]
        var_prior = - np.sum(params[:, -8:] * params[:, -8:])
        variances = np.exp(params[:,-8:]) # axis aligned variances
        ll = 0.
        for i in xrange(T.shape[0]):
            ll = ll + np.sum(
                gmm_util.mog_loglike(
                    T[i],
                    means = means[i,:,None].T,
                    icovs = np.array([ np.diag(1./variances[i]) ]),
                    dets  = np.array([1.]),
                    pis   = np.array([1.]))
                )
        return - log_prior - ll - var_prior

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(pred_fun(W_vect, X), axis=1))

    return parser.N, predict_distribution, loss, frac_err


#############################################################################
# Layer classes
#############################################################################
class conv_layer(object):
    def __init__(self, kernel_shape, num_filters):
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters

    def forward_pass(self, inputs, param_vector):
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        conv = convolve(inputs, params, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]), mode='valid')
        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_shape[0], self.num_filters)
                                          + self.kernel_shape)
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape(input_shape[1:], self.kernel_shape)
        print "Conv layer: ", input_shape, "=>", output_shape
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)

class fast_conv_layer(object):
    def __init__(self, kernel_shape, num_filters, img_shape):
        self.kernel_shape = kernel_shape
        self.num_filters  = num_filters
        self.img_shape    = img_shape

    def forward_pass(self, inputs, param_vector):
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        if len(inputs.shape) == 4:
            inputs = make_img_col(inputs)
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        conv = np.zeros((inputs.shape[0], params.shape[0]) + \
                        self.conv_output_shape(self.img_shape, self.kernel_shape))
        for k in range(self.num_filters):
            for i in range(inputs.shape[0]):
                conv[i, k, :, :] = \
                    convolve_im2col(inputs[i,:,:],
                                    params[k,:,:,:],
                                    block_size = self.kernel_shape,
                                    skip=1,
                                    orig_img_shape = self.img_shape)

        # conv out is [num_data, num_filters, y, x]
        #conv = convolve(inputs, params, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]), mode='valid')

        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (self.num_filters,
                                           self.kernel_shape[0],
                                           self.kernel_shape[1],
                                           input_shape[0]))
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape(input_shape[1:], self.kernel_shape)
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)


class maxpool_layer(object):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape

    def build_weights_dict(self, input_shape):
        # input_shape dimensions: [color, y, x]
        output_shape = list(input_shape)
        for i in [0, 1]:
            assert input_shape[i + 1] % self.pool_shape[i] == 0, \
                "maxpool shape should tile input exactly"
            output_shape[i + 1] = input_shape[i + 1] / self.pool_shape[i]

        print "Max pool layer: ", input_shape, "=>", output_shape
        return 0, output_shape

    def forward_pass(self, inputs, param_vector):
        new_shape = inputs.shape[:2]
        for i in [0, 1]:
            pool_width = self.pool_shape[i]
            img_width = inputs.shape[i + 2]
            new_shape += (pool_width, img_width / pool_width)
        result = inputs.reshape(new_shape)
        return np.max(np.max(result, axis=2), axis=3)

class full_layer(object):
    def __init__(self, size):
        self.size = size

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_size, self.size))
        self.parser.add_weights('biases', (self.size,))
        print "full layer: ", input_shape, "=>", (self.size, )
        return self.parser.N, (self.size,)

    def forward_pass(self, inputs, param_vector):
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

class linear_layer(full_layer):
    def nonlinearity(self, x):
        return x

class tanh_layer(full_layer):
    def nonlinearity(self, x):
        return np.tanh(x)

class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - logsumexp(x, axis=1, keepdims=True)


############################################################################
# Util funcs
############################################################################

def gauss_filt_2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    import numpy as np
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def logsumexp(X, axis, keepdims=False):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))



#############################################################################
# Funcs for fast convolutions: 
#   - represent image and weights in column format
#     and use im2col based convolution (matrix multiply)
#############################################################################
def make_img_col(imgs, filter_shape=(5,5), verbose=False):
    """ Takes a stack of images with shape [N, color, y, x]
        outputs a column stack of images with shape
        [N, filter_x*filter_y, conv_out_y*conv_out_x]
    """
    imgs     = np.rollaxis(imgs, 1, 4)
    img0     = im2col(imgs[0, :, :, :], filter_shape)
    col_imgs = np.zeros((imgs.shape[0], img0.shape[0], img0.shape[1]))
    for i, img in enumerate(imgs):
        if i % 5000 == 0 and verbose:
            print "%d of %d"%(i, len(imgs))
        col_imgs[i,:,:] = im2col(img, filter_shape)
    return col_imgs


def im2col(img, block_size = (5, 5), skip = 1):
    """ stretches block_size size'd patches centered skip distance 
        away in both row/column space, stacks into columns (and stacks)
        bands into rows

        Use-case is for storing images for quick matrix multiplies
           - blows up memory usage by quite a bit (factor of 10!)

        motivated by implementation discussion here: 
            http://cs231n.github.io/convolutional-networks/

        edited from snippet here:
            http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    """
    # stack depth bands (colors)
    if len(img.shape) == 3:
        return np.vstack([ im2col(img[:,:,k], block_size, skip)
                           for k in xrange(img.shape[2]) ])

    # input array and block size
    A = img
    B = block_size

    # Parameters
    M,N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(0, row_extent, skip)[:,None]*N + np.arange(0, col_extent, skip)

    # Get all actual indices & index into input array for final output
    out = np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())
    return out

def convolve_im2col(img_cols, filt, block_size, skip, orig_img_shape):
    """ convolves an image already in the column representation
        with a filter bank (not in the column representation)
    """
    filtr = im2col(filt, block_size=block_size, skip=skip)
    out_num_rows = (orig_img_shape[0] - block_size[0])/skip + 1
    out_num_cols = (orig_img_shape[1] - block_size[1])/skip + 1
    outr = np.dot(filtr.T, img_cols)
    out  = np.reshape(outr, (out_num_rows, out_num_cols))
    return out


if __name__ == '__main__':

    #skip = 1
    #block_size = (11, 11)
    #img = np.random.randn(227, 227, 3)
    #filt = np.dstack([gauss_filt_2D(shape=block_size,sigma=2) for k in range(3)])
    #img_cols = im2col(img, block_size=block_size, skip=skip)
    #out = convolve_im2col(img_cols, filt, block_size, skip, img.shape)

    # Network parameters
    L2_reg = 1.0
    input_shape = (1, 28, 28)
    layer_specs = [fast_conv_layer((5, 5), 6, input_shape[1:]),
                   #conv_layer((5, 5), 6),
                   maxpool_layer((2, 2)),
                   #conv_layer((5, 5), 16),
                   fast_conv_layer((5, 5), 16, (12, 12)),
                   maxpool_layer((2, 2)),
                   tanh_layer(120),
                   tanh_layer(84),
                   softmax_layer(10)]

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 256
    num_epochs = 25

    # Load and process MNIST data (borrowing from Kayak)
    print("Loading training data...")
    import imp, urllib
    add_color_channel = lambda x : x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    #source, _ = urllib.urlretrieve(
    #    'https://raw.githubusercontent.com/HIPS/Kayak/master/examples/data.py')
    #data = imp.load_source('data', source).mnist()
    train_images, train_labels, test_images, test_labels = data
    train_images = add_color_channel(train_images) / 255.0
    test_images  = add_color_channel(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    #train_cols = make_img_col(train_images)
    #test_cols  = make_img_col(test_images)

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
    loss_grad = grad(loss_fun)

    # test loss
    #loss_fun(W, train_images[:20], train_labels[:20])
    loss_fun(W, train_cols[:50], train_labels[:50])
    assert False

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    #quick_grad_check(loss_fun, W, (train_images[:50], train_labels[:50]))

    print("    Epoch      |    Train err  |   Test error  ")
    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_images, test_labels)
        train_perf = frac_err(W, train_images, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs = make_batches(N_data, batch_size)
    cur_dir = np.zeros(N_weights)

    for epoch in range(num_epochs):
        print_perf(epoch, W)
        for idxs in batch_idxs:
            grad_W = loss_grad(W, train_images[idxs], train_labels[idxs])
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir

