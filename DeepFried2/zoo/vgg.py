import DeepFried2 as df

import scipy.io
import numpy as np


def model_head(fully_conv=True):
    if fully_conv:
        return [
            df.SpatialConvolutionCUDNN( 512, 4096, (7,7), border='valid'), df.ReLU(),
            df.Dropout(0.5),
            df.SpatialConvolutionCUDNN(4096, 4096, (1,1), border='valid'), df.ReLU(),
            df.Dropout(0.5),
            df.SpatialConvolutionCUDNN(4096, 1000, (1,1), border='valid'),
            df.SpatialSoftMaxCUDNN(),
        ]
    else:
        return [
            df.Reshape(-1, 512*7*7),
            df.Linear(512*7*7, 4096), df.ReLU(),
            df.Dropout(0.5),
            df.Linear(4096, 4096), df.ReLU(),
            df.Dropout(0.5),
            df.Linear(4096, 1000),
            df.SoftMax()
        ]


def params(large=True, fully_conv=True, fname=None):
    # Thanks a lot to @317070 (Jonas Degrave) for this!
    if large:
        fname = fname or df.zoo.download('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', saveas='vgg19-imagenet.mat', desc='vgg19-imagenet.mat')
        layers = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34,37,39,41]
    else:
        fname = fname or df.zoo.download('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', saveas='vgg16-imagenet.mat', desc='vgg16-imagenet.mat')
        layers = [0,2,5,7,10,12,14,17,19,21,24,26,28,31,33,35]

    params = []

    mat = scipy.io.loadmat(fname)
    for l in layers:
        W = mat['layers'][0,l][0,0][2][0,0]
        W = W.transpose(3,2,0,1)
        b = mat['layers'][0,l][0,0][2][0,1]
        b = b.squeeze()
        params += [W, b]

    # For the "classic" case of fully-connected layers as GEMM, we need to
    # reshape the parameters into the matrices they are.
    if not fully_conv:
        params[-6] = params[-6].reshape(4096, -1).T
        params[-4] = params[-4].squeeze().T
        params[-2] = params[-2].squeeze().T

    # The mean is actually a single scalar per color channel.
    # mean = mat['normalization'][0,0][0]  # This is H,W,C
    # mean = np.mean(mean, axis=(0,1))
    # The format seems to have changed some time.
    mean = mat['meta']['normalization'][0,0][0,0]['averageImage'][0,0]

    # Note; there's also the key `description` which we could use as human-readable name.
    classes = np.array([cls[0] for cls in mat['meta']['classes'][0,0][0]['name'][0][0,:]])

    return params, mean, classes
