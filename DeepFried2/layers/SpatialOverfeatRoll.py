import DeepFried2 as df


from theano.sandbox.cuda.basic_ops import gpu_contiguous


class PyCudaOp(df.th.sandbox.cuda.GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self)) + 2

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, inp, dy, dx):
        inp = df.th.sandbox.cuda.as_cuda_ndarray_variable(inp)
        dy = df.th.gof.Constant(df.th.scalar.int32, dy)
        dx = df.th.gof.Constant(df.th.scalar.int32, dx)
        return df.th.Apply(self, [inp, dy, dx], [inp.type()])


class RollOpBase(PyCudaOp):
    def c_support_code(self):
        c_support_code = """
            __global__ void roll(const float *input, float *output, const int B, const int C, const int H, const int W, const int dy, const int dx)
        {
            // (B,C,H,W) are the dimensions.
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int b = blockIdx.y * blockDim.y + threadIdx.y;
            const int c =  i / (H*W);
            const int y = (i % (H*W)) / W;
            const int x =  i % W;
            const int y_out = y/dy;
            const int x_out = x/dx;
            const int b_out = b*dy*dx + dx*(y%dy) + (x%dx);
            if (b < B && c < C && y < H && x < W)
            {
                output[b_out * C * ((H+1)/dy) * ((W+1)/dx) +
                               c * ((H+1)/dy) * ((W+1)/dx) +
                                        y_out * ((W+1)/dx) +
                                                     x_out
                ] = input[b * C * H * W +
                              c * H * W +
                                  y * W +
                                      x];
            }
        }
        """
        return c_support_code

    def c_code(self, node, name, inputs, outputs, sub):
        fail = sub['fail']

        assert len(inputs) == 3, str(type(self)) + " only takes three inputs."
        assert len(outputs) == 1, str(type(self)) + " only takes one output."
        inp, dy, dx = inputs
        out, = outputs

        c_code = """
        {
            const int nbatch = CudaNdarray_HOST_DIMS(%(inp)s)[0];
            const int nfeats = CudaNdarray_HOST_DIMS(%(inp)s)[1];
            const int height = CudaNdarray_HOST_DIMS(%(inp)s)[2];
            const int width  = CudaNdarray_HOST_DIMS(%(inp)s)[3];

            int out_shape[] = {nbatch * %(dy)s * %(dx)s, nfeats, (height+1) / %(dy)s, (width+1) / %(dx)s};
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
            dim3 grid((int)(ceil(((float)nfeats * height * width) / block.x)),
                      (int)(ceil(((float)nbatch) / block.y)),
                       1);

            roll<<<grid, block>>>(CudaNdarray_DEV_DATA(%(inp)s),
                                  CudaNdarray_DEV_DATA(%(out)s),
                                  nbatch, nfeats, height, width, %(dy)s, %(dx)s);

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


class UnRollOpBase(PyCudaOp):
    def c_support_code(self):
        c_support_code = """
        __global__ void unroll(const float *input, float *output, const int B, const int C, const int H, const int W, const int dy, const int dx)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int b = blockIdx.y * blockDim.y + threadIdx.y;
            const int c =  i / (W*H);
            const int y = (i % (W*H)) / W;
            const int x =  i % W;
            const int y_out = y*dy + ((b/dx) % dy);
            const int x_out = x*dx + ( b     % dx);
            const int b_out = b/(dy*dx);
            if (b < B && c < C)
            {
                output[b_out * C * H*dy * W*dx +
                               c * H*dy * W*dx +
                                  y_out * W*dx +
                                         x_out
                ] = input[b * C * H * W +
                              c * H * W +
                                  y * W +
                                      x];
            }
        }
        """

        return c_support_code

    def c_code(self, node, name, inputs, outputs, sub):
        fail = sub['fail']

        assert len(inputs) == 3, str(type(self)) + " only takes three inputs."
        assert len(outputs) == 1, str(type(self)) + " only takes one output."
        inp, dy, dx = inputs
        out, = outputs

        c_code = """
        {
            const int nbatch = CudaNdarray_HOST_DIMS(%(inp)s)[0];
            const int nfeats = CudaNdarray_HOST_DIMS(%(inp)s)[1];
            const int height = CudaNdarray_HOST_DIMS(%(inp)s)[2];
            const int width  = CudaNdarray_HOST_DIMS(%(inp)s)[3];

            int out_shape[] = {nbatch/(%(dy)s * %(dx)s), nfeats, height*%(dy)s, width*%(dx)s};
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
            dim3 grid((int)(ceil(((float)nfeats * height * width) / block.x)),
                      (int)(ceil(((float)nbatch) / block.y)),
                       1);

            unroll<<<grid, block>>>(CudaNdarray_DEV_DATA(%(inp)s),
                                    CudaNdarray_DEV_DATA(%(out)s),
                                    nbatch, nfeats, height, width, %(dy)s, %(dx)s);

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


class RollOpGrad(UnRollOpBase):
    pass


class UnRollOpGrad(RollOpBase):
    pass


class RollOp(RollOpBase):
    def grad(self, inp, grads):
        top, = grads
        top = df.th.sandbox.cuda.basic_ops.gpu_contiguous(top)
        return [RollOpGrad()(top)]


class UnRollOp(UnRollOpBase):
    def grad(self, inp, grads):
        top, = grads
        top = df.th.sandbox.cuda.basic_ops.gpu_contiguous(top)
        return [UnRollOpGrad()(top)]

roll = RollOp()
unroll = UnRollOp()


class SpatialOverfeatRoll(df.Module):
    def __init__(self, dy=2, dx=2):
        df.Module.__init__(self)
        self.d = (dy, dx)

    def symb_forward(self, symb_input):
        return roll(symb_input, *self.d)


class SpatialOverfeatUnroll(df.Module):
    def __init__(self, dy=2, dx=2):
        df.Module.__init__(self)
        self.d = (dy, dx)

    def symb_forward(self, symb_input):
        return unroll(symb_input, *self.d)
