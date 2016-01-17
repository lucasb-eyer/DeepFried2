import DeepFried2 as df
from . import vgg as _vgg


def model(fully_conv=True):
    conv3 = lambda nin, nout: df.SpatialConvolutionCUDNN(nin, nout, (3,3), border='same')

    return df.Sequential(
        conv3(  3, 64), df.ReLU(),
        conv3( 64, 64), df.ReLU(),
        df.SpatialMaxPoolingCUDNN((2,2)),
        conv3( 64,128), df.ReLU(),
        conv3(128,128), df.ReLU(),
        df.SpatialMaxPoolingCUDNN((2,2)),
        conv3(128,256), df.ReLU(),
        conv3(256,256), df.ReLU(),
        conv3(256,256), df.ReLU(),
        conv3(256,256), df.ReLU(),
        df.SpatialMaxPoolingCUDNN((2,2)),
        conv3(256,512), df.ReLU(),
        conv3(512,512), df.ReLU(),
        conv3(512,512), df.ReLU(),
        conv3(512,512), df.ReLU(),
        df.SpatialMaxPoolingCUDNN((2,2)),
        conv3(512,512), df.ReLU(),
        conv3(512,512), df.ReLU(),
        conv3(512,512), df.ReLU(),
        conv3(512,512), df.ReLU(),
        df.SpatialMaxPoolingCUDNN((2,2)),
        *_vgg.model_head(fully_conv)
    )


def params(fully_conv=True, fname=None):
    return _vgg.params(large=True, fully_conv=fully_conv, fname=fname)


def pretrained(fully_conv=True, fname=None):
    # Create the model
    vgg = model(fully_conv)

    # Load the parameters and a few more settings from the downloaded pretrained file.
    values, mean, classes = params(fully_conv, fname)

    # Load the pretrained parameter values into the network.
    for p, v in zip(vgg.parameters(), values):
        p.set_value(v)

    return vgg, mean, classes
