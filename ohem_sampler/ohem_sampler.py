# coding: utf-8
# pylint: disable= not-callable
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import mxnet as mx

class OHEMSamplerOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, neg_ratio, batch_ohem):
        super(OHEMSamplerOperator, self).__init__()
        self.num_classes = num_classes
        self.neg_ratio = neg_ratio
        self.batch_ohem = batch_ohem

    def forward(self, is_train, req, in_data, out_data, aux):
        # conf_data, target, weight  
        conf_d = in_data[0].asnumpy()   # (B, N, C)
        conf_t = in_data[1].asnumpy()   # (B, N)
        conf_w = in_data[2].asnumpy()   # (B, N, 1)

        # firstly, set all bg weight to 0
        conf_w[conf_t == 0] = 0

        bg_prob = conf_d[:,:,0]   # (B, N)
        bg_prob[conf_t != 0] = 1.0  # set non-bg prob to 1
        neg_nums = self.neg_ratio * np.sum(conf_t > 0, axis=-1)  # (B,)
        neg_nums = np.ceil(neg_nums).astype(np.int32)

        if self.batch_ohem:
            neg_nums = np.sum(neg_nums).astype(np.int32)     # (1,)
            bg_idx = conf_t.flatten() == 0
            bg_prob_f = bg_prob.flatten()
            bgsort_idx = np.argsort(bg_prob_f[bg_idx], axis=-1)
            conf_w_f = conf_w.flatten()
            conf_w_f[bg_idx][bgsort_idx[:neg_nums]] = 1
            conf_w = conf_w_f.reshape((conf_d.shape[0], -1, 1))
        else:
            for batch_id in range(bg_prob.shape[0]):
                bg_idx = conf_t[batch_id].flatten() == 0
                bgsort_idx = np.argsort(bg_prob[batch_id][bg_idx], axis=-1)
                pick_idx = bgsort_idx[:neg_nums[batch_id]]
                conf_w[batch_id][bg_idx][pick_idx] = 1

        self.assign(out_data[0], req[0], conf_w)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('ohem_sampler')
class OHEMLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, neg_ratio=3.0, batch_ohem=True):
        super(OHEMLossProp, self).__init__(need_top_grad=False)
        self.num_classes = float(num_classes)
        self.neg_ratio = float(neg_ratio)
        assert batch_ohem in ['True', 'False']
        self.batch_ohem = batch_ohem == 'True'

    def list_arguments(self):
        return ['conf_data', 'conf_target', 'conf_weight']

    def list_outputs(self):
        return ['new_conf_weight',]

    def infer_shape(self, in_shape):
        # in_shape [(B, N, C), (B, N), (B, N)]
        # out_shape [(B,)]
        assert len(in_shape) == 3
        out_shape = []
        out_shape.append(in_shape[-1])
        return in_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return OHEMSamplerOperator(self.num_classes, self.neg_ratio, self.batch_ohem)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
