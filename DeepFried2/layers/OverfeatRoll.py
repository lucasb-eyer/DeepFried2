import DeepFried2 as df


from theano.sandbox.cuda import CudaNdarrayType, GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable, gpu_contiguous)


class PyCudaOp(GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, inp):
        inp = as_cuda_ndarray_variable(inp)
        return df.th.Apply(self, [inp], [inp.type()])


class RollOpBase(PyCudaOp):
    def c_support_code(self):
        c_support_code = """
            __global__ void maxpool_roll(float *input, float *output, int batch_size, int feature_size, int height_size, int width_size)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int batch = blockIdx.y * blockDim.y + threadIdx.y;
            int map_size = height_size * width_size;
            int feature = x / map_size;
            int height = (x % map_size) / width_size;
            int width = x % width_size;
            int height_out = height / 2;
            int width_out = width / 2;
            int batch_out = batch * 4;
            if (height % 2 == 0 && width % 2 == 1)
            {
                batch_out += 1;
            }
            else if (height % 2 == 1 && width % 2 == 0)
            {
                batch_out += 2;
            }
            else if (height % 2 == 1 && width % 2 == 1)
            {
                batch_out += 3;
            }
            if (batch < batch_size && feature < feature_size && height_out * 2 < height_size && width_out * 2 < width_size)
            {
                output[batch_out * (feature_size * ((height_size + 1) / 2) * ((width_size + 1) / 2)) +
                       feature * (((height_size + 1) / 2) * ((width_size + 1) / 2)) +
                       height_out * ((width_size + 1) / 2) +
                       width_out] = input[batch * (feature_size * height_size * width_size) +
                                          feature * (height_size * width_size) +
                                          height * width_size +
                                          width];
            }
        }
        """
        return c_support_code

    def c_code(self, node, name, inputs, outputs, sub):
        fail = sub['fail']

        inp, = inputs
        out, = outputs

        c_code = """
        {
            int batch_size = CudaNdarray_HOST_DIMS(%(inp)s)[0];
            int n_features = CudaNdarray_HOST_DIMS(%(inp)s)[1];
            int height = CudaNdarray_HOST_DIMS(%(inp)s)[2];
            int width = CudaNdarray_HOST_DIMS(%(inp)s)[3];

            int out_shape[] = {batch_size * 4, n_features, (height + 1) / 2, (width + 1) / 2};
            if (NULL == %(out)s || CudaNdarray_NDIM(%(inp)s) != CudaNdarray_NDIM(%(out)s) ||
                                    !(CudaNdarray_HOST_DIMS(%(out)s)[0] == out_shape[0] &&
                                     CudaNdarray_HOST_DIMS(%(out)s)[1] == out_shape[1] &&
                                     CudaNdarray_HOST_DIMS(%(out)s)[2] == out_shape[2] &&
                                     CudaNdarray_HOST_DIMS(%(out)s)[3] == out_shape[3]))
            {
                Py_XDECREF(%(out)s);
                %(out)s = (CudaNdarray*)CudaNdarray_ZEROS(CudaNdarray_NDIM(%(inp)s), out_shape);
            }

            if (!%(out)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }

            dim3 block(16, 16, 1);
            dim3 grid((int)(ceil(((float)n_features * height * width) / block.x)),
                      (int)(ceil(((float)batch_size) / block.y)),
                       1);

            maxpool_roll<<<grid, block>>>(CudaNdarray_DEV_DATA(%(inp)s),
                                          CudaNdarray_DEV_DATA(%(out)s),
                                          batch_size, n_features, height, width);

            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, cudaGetErrorString(sts));
                %(fail)s;
            }
        }
        """
        return c_code % locals()


class RollOp(RollOpBase):
    def grad(self, inp, grads):
        top, = grads
        top = gpu_contiguous(top)
        return [RollOpGrad()(top)]


class UnRollOpBase(PyCudaOp):
    def c_support_code(self):
        c_support_code = """
        __global__ void maxpool_unroll(float *input, float *output, int batch_size, int feature_size, int height_size, int width_size)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int batch = blockIdx.y * blockDim.y + threadIdx.y;
            int map_size = height_size * width_size;
            int feature = x / map_size;
            int height = (x % map_size) / width_size;
            int width = x % width_size;
            int height_out = height * 2;
            int width_out = width * 2;
            int batch_out = batch / 4;
            if (batch % 4 == 1)
            {
                width_out += 1;
            }
            else if (batch % 4 == 2)
            {
                height_out += 1;
            }
            else if (batch % 4 == 3)
            {
                height_out += 1;
                width_out += 1;
            }
            if (batch < batch_size && feature < feature_size)
            {
                output[batch_out * (feature_size * height_size * 2 * width_size * 2) +
                       feature * (height_size * 2 * width_size * 2) +
                       height_out * width_size * 2 +
                       width_out] = input[batch * (feature_size * height_size * width_size) +
                                          feature * (height_size * width_size) +
                                          height * width_size +
                                          width];
            }
        }
        """

        return c_support_code

    def c_code(self, node, name, inputs, outputs, sub):
        fail = sub['fail']

        inp, = inputs
        out, = outputs

        c_code = """
        {
            int batch_size = CudaNdarray_HOST_DIMS(%(inp)s)[0];
            int n_features = CudaNdarray_HOST_DIMS(%(inp)s)[1];
            int height = CudaNdarray_HOST_DIMS(%(inp)s)[2];
            int width = CudaNdarray_HOST_DIMS(%(inp)s)[3];

            int out_shape[] = {batch_size / 4, n_features, height * 2, width * 2};
            if (NULL == %(out)s || CudaNdarray_NDIM(%(inp)s) != CudaNdarray_NDIM(%(out)s) ||
                                   !(CudaNdarray_HOST_DIMS(%(out)s)[0] == out_shape[0] &&
                                     CudaNdarray_HOST_DIMS(%(out)s)[1] == out_shape[1] &&
                                     CudaNdarray_HOST_DIMS(%(out)s)[2] == out_shape[2] &&
                                     CudaNdarray_HOST_DIMS(%(out)s)[3] == out_shape[3]))
            {
                Py_XDECREF(%(out)s);
                %(out)s = (CudaNdarray*)CudaNdarray_NewDims(CudaNdarray_NDIM(%(inp)s), out_shape);
            }

            if (!%(out)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }

            dim3 block(16, 16, 1);
            dim3 grid((int)(ceil(((float)n_features * height * width) / block.x)),
                      (int)(ceil(((float)batch_size) / block.y)),
                       1);

            maxpool_unroll<<<grid, block>>>(CudaNdarray_DEV_DATA(%(inp)s),
                                            CudaNdarray_DEV_DATA(%(out)s),
                                            batch_size, n_features, height, width);

            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, cudaGetErrorString(sts));
                %(fail)s;
            }
        }
        """
        return c_code % locals()


class UnRollOp(UnRollOpBase):
    def grad(self, inp, grads):
        top, = grads
        top = gpu_contiguous(top)
        return [UnRollOpGrad()(top)]


class RollOpGrad(UnRollOpBase):
    pass


class UnRollOpGrad(RollOpBase):
    pass

unroll = UnRollOp()
roll = RollOp()


class OverfeatRoll(df.Module):
    def __init__(self):
        df.Module.__init__(self)

    def symb_forward(self, symb_input):
        return roll(symb_input)


class OverfeatUnroll(df.Module):
    def __init__(self):
        df.Module.__init__(self)

    def symb_forward(self, symb_input):
        return unroll(symb_input)
