import mxnet as mx
import numpy as np

class WingOperator(mx.operator.CustomOp):
    def __init__(self, w, eps):
        super(WingOperator, self).__init__()
        self._w = float(w)
        self._eps = float(eps)
        assert self._w >= 0

    def forward(self, is_train, req, in_data, out_data, aux):
        out = mx.nd.zeros_like(in_data[0])
        sign = ((in_data[0] >= 0) - 0.5) * 2

        case1 = (in_data[0] > -self._w) * (in_data[0] < self._w)
        value1 = self._w * mx.nd.log(1 + sign * in_data[0] / self._eps)
        out = mx.nd.where(case1, value1, out)

        case2 = mx.nd.ones_like(in_data[0]) - case1
        value2 = sign * in_data[0] - (self._w - self._w * np.log(1 + self._w / self._eps))
        out = mx.nd.where(case2, value2, out)

        self.assign(out_data[0], req[0], out)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        out = mx.nd.zeros_like(in_data[0])
        sign = ((in_data[0] >= 0) - 0.5) * 2

        case1 = (in_data[0] > -self._w) * (in_data[0] < self._w)
        value1 = sign * self._w / (self._eps + sign * in_data[0])
        out = mx.nd.where(case1, value1, out)

        case2 = mx.nd.ones_like(in_data[0]) - case1
        out = mx.nd.where(case2, sign, out)

        # grad for data
        out = out*out_grad[0]
        self.assign(in_grad[0], req[0], out)

@mx.operator.register('wing')
class WingProp(mx.operator.CustomOpProp):
    def __init__(self, w=10.0, eps=2.0):
        super(WingProp, self).__init__()
        self._w = float(w)
        self._eps = float(eps)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return WingOperator(self._w, self._eps)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        deps = []
        if self.need_top_grad_:
            deps.extend(out_grad)
        deps.extend(in_data)
        deps.extend(out_data)
        return deps

def wing_loss(data, w=10.0, eps=2.0):
    return mx.sym.Custom(data=data, w=10.0, eps=2.0, op_type='wing')

