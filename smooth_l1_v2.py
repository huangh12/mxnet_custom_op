import mxnet as mx
import numpy as np


class SmoothL1V2Operator(mx.operator.CustomOp):
    def __init__(self):
        super(SmoothL1V2Operator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        sigma_sqr = in_data[1].reshape((-1))[0] ** 2
        isigma_sqr = 1.0 / sigma_sqr
        out = mx.nd.zeros_like(in_data[0])

        case1 = in_data[0] > isigma_sqr
        value1 = in_data[0] - 0.5 * isigma_sqr
        out = mx.nd.where(case1, value1, out)

        case2 = in_data[0] < -isigma_sqr
        value2 = - in_data[0] - 0.5 * isigma_sqr
        out = mx.nd.where(case2, value2, out)

        case3 = mx.nd.ones_like(in_data[0]) - case1 - case2
        value3 = 0.5 * in_data[0] ** 2 * sigma_sqr
        out = mx.nd.where(case3, value3, out)

        self.assign(out_data[0], req[0], out)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        sigma_sqr = in_data[1].reshape((-1))[0] ** 2
        isigma_sqr = 1.0 / sigma_sqr

        out1 = mx.nd.zeros_like(in_data[0])

        case1 = in_data[0] > isigma_sqr
        out1 = mx.nd.where(case1, mx.nd.ones_like(out1), out1)

        case2 = in_data[0] < -isigma_sqr
        out1 = mx.nd.where(case2, -1 * mx.nd.ones_like(out1), out1)

        case3 = mx.nd.ones_like(in_data[0]) - case1 - case2
        value3 = in_data[0] * sigma_sqr
        out1 = mx.nd.where(case3, value3, out1)

        # grad for data
        self.assign(in_grad[0], req[0], out1*out_grad[0])
        # grad for sigma (not learn)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('smooth_l1_v2')
class SmoothL1V2Prop(mx.operator.CustomOpProp):
    def __init__(self):
        super(SmoothL1V2Prop, self).__init__()

    def list_arguments(self):
        return ['data', 'sigma']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[0]], [in_shape[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return SmoothL1V2Operator()


def smooth_l1_v2(data, sigma):
    return mx.sym.Custom(data=data, sigma=sigma,
                         op_type='smooth_l1_v2')

