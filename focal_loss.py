import mxnet as mx

class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self, grad_scale=1.0, alpha=0.25, gamma=2.0, normalize='positive', use_ignore=False, ignore_label=-1, anchor_loss=False):
        super(FocalLossOperator, self).__init__()
        self._grad_scale = grad_scale
        self._use_ignore = use_ignore
        self._ignore_label = ignore_label
        self._anchor_loss = anchor_loss
        self._alpha = alpha
        self._gamma = gamma
        self._normalize = normalize

    def forward(self, is_train, req, in_data, out_data, aux):
        # CE loss output
        # self.assign(out_data[0], req[0], mx.nd.divide(1.0, (1.0 + mx.nd.exp(-in_data[0]))))

        
        # Focal loss output (to check numerical graident)
        label = in_data[1]
        prob = mx.nd.divide(1.0, (1.0 + mx.nd.exp(-in_data[0])))

        out_loss = mx.nd.zeros_like(prob)
        pos_keep = label > 0
        fg_loss = - self._alpha * (1 - prob)**self._gamma * mx.nd.log(prob)
        out_loss = mx.nd.where(pos_keep, fg_loss, out_loss)
        neg_keep = label == 0
        bg_loss = - (1-self._alpha) * (prob)**self._gamma * mx.nd.log(1-prob)
        out_loss = mx.nd.where(neg_keep, bg_loss, out_loss)

        if self._use_ignore:
            keep = label != self._ignore_label
            if self._normalize == 'valid':
                num_output = mx.nd.sum(keep)
            elif self._normalize == 'positive':
                num_output = mx.nd.sum((label > 0))
        else:
            raise ValueError('unimplemented')            

        if num_output == 0:
            self.assign(out_data[0], req[0], 0.)
        else:
            out_loss /= num_output
            self.assign(out_data[0], req[0], out_loss)
        

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        bottom_data = in_data[0]
        label = in_data[1]
        # pred = out_data[0]
        pred = mx.nd.divide(1.0, (1.0 + mx.nd.exp(-in_data[0])))
        if self._use_ignore:
            keep = label != self._ignore_label
            if self._normalize == 'valid':
                num_output = mx.nd.sum(keep)
            elif self._normalize == 'positive':
                num_output = mx.nd.sum((label > 0))
        else:
            raise ValueError('unimplemented')

        if num_output == 0:
            self.assign(in_grad[0], req[0], 0.)
        else:
            # data
            # grad_scale = self._grad_scale / float(num_output)
            grad_scale = self._grad_scale / num_output
            pos_keep = label > 0
            neg_keep = label == 0

            # (1-p)**g * (1 - p - g*p*log(p))
            # tmp = - self._alpha * ((1-pred)**self._gamma * (1-pred-self._gamma*pred*mx.nd.log(mx.nd.max(pred, sys.float_info.min))))

            tmp = - self._alpha * ((1-pred)**self._gamma * (1-pred-self._gamma*pred*mx.nd.log(pred)))
            grad_data = mx.nd.where(pos_keep, tmp, mx.nd.zeros_like(pred))

            # (p**g) * (g*(1-p)*log(1-p) - p)
            tmp = - (1-self._alpha) * (pred**self._gamma) * (self._gamma * (1-pred) * mx.nd.log(1-pred) - pred) 
            grad_data = mx.nd.where(neg_keep, tmp, grad_data)
            
            # It's weird the below impl.(similar to cuda impl.) cannot pass gradient check?
            # tmp = (bottom_data >= 0) * neg_keep
            # tmp_value = - (1-self._alpha) * (pred ** self._gamma) * (-bottom_data - mx.nd.log(1+mx.nd.exp(-bottom_data)) * (1-pred)*self._gamma - pred)
            # grad_data = mx.nd.where(tmp, tmp_value, grad_data)
            # tmp = (bottom_data < 0) * neg_keep
            # tmp_value = - (1-self._alpha) * (pred ** self._gamma) * (-mx.nd.log(1+mx.nd.exp(bottom_data)) * (1-pred)*self._gamma - pred)
            # grad_data = mx.nd.where(tmp, tmp_value, grad_data)

            grad_data *= grad_scale
            self.assign(in_grad[0], req[0], grad_data)

        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('focal_loss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, alpha=0.25, gamma=2.0, normalize='positive', use_ignore=False, ignore_label=-1, anchor_loss=False):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self._grad_scale = float(grad_scale)
        self._use_ignore = use_ignore == str(True)
        self._ignore_label = int(ignore_label)
        self._anchor_loss = anchor_loss == str(True)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._normalize = str(normalize)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[0]], [in_shape[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._grad_scale, self._alpha, self._gamma, self._normalize, self._use_ignore, self._ignore_label, self._anchor_loss)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        deps = []
        deps.extend(in_data)
        deps.extend(out_data)
        return deps


def focal_loss(data, label, name=None, grad_scale=1.0, alpha=0.25, gamma=2.0, normalize='positive', use_ignore=False, ignore_label=-1):
    return mx.sym.Custom(data=data, label=label, name=name,
                         op_type='focal_loss',
                         grad_scale=grad_scale,
                         alpha=alpha,
                         gamma=gamma,
                         normalize=normalize,
                         use_ignore=use_ignore,
                         ignore_label=ignore_label)

