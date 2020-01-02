This repo collects the mxnet custom operators implemented/modified by me.


## Gradient check ##

[check_op.py](check_op.py) is to check the consistency of numerical gradients and symbolic gradients.
Note the [python/mxnet/test_utils.py](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/test_utils.py#L981) only supports gradient check of operator with need of `top_grad`. 
For operator with no need of `top_grad` (such as many loss layers), it should be modified such that the **__random_proj** of below lines
are identity mapping. Otherwise, your gradient check will never pass through. This can be a implicit bug of MXNet.

```Python
    ...
    input_shape = {k: v.shape for k, v in location.items()}
    _, out_shape, _ = sym.infer_shape(**input_shape)
    proj = mx.sym.Variable("__random_proj")
    out = sym * proj
    out = mx.sym.make_loss(out)

    location = dict(list(location.items()) +
                    [("__random_proj", mx.nd.array(random_projection(out_shape[0]),
                                                   ctx=ctx, dtype=dtype))])
    args_grad_npy = dict([(k, np.random.normal(0, 0.01, size=location[k].shape))
                          for k in grad_nodes]
                         + [("__random_proj", np.random.normal(0, 0.01, size=out_shape[0]))])
    ...
```

## operators intro ## 

It currently includes:

1. focal loss
   
   [ndarray version](focal_loss.py), which support positive(the focal loss paper used) or valid loss normalization

   [cuda version](sigmoid_focal_loss_v2-inl.h), which supports dymanic grad scale, i.e., grad_scale is a variable input to the operator instead of a fixed parameter. This op can be used in Cascade RetinaNet.

1. smooth_l1_v2
   
   [ndarray version](smooth_l1_v2.py). The difference between the normal smooth_l1 loss is that, its sigma is a variable to pass into the op.
