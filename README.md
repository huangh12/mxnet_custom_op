This repo collects the mxnet custom operators implemented/modified by myself.


## Gradient check ##

**Be careful of trap in gradient check.** In this repo, the [check_op.py](check_op.py) is to check the consistency of numerical gradients and symbolic gradients of the MXNet operators. It actually invokes the `check_numeric_gradient` function in [python/mxnet/test_utils.py](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/test_utils.py#L981). However, there is an implicit bug in `check_numeric_gradient` that it only supports gradient check of operators in need of `top_grad`. 
For operators without need of `top_grad` (such as many loss layers), you should modify the [python/mxnet/test_utils.py](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/test_utils.py#L981) such that the **__random_proj** is an identity mapping(see below lines). Otherwise, your gradient check will never pass through.

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

## operator list ## 

1. **Focal Loss v2**
   
   [NDArray version](focal_loss_v2/focal_loss.py). It supports positive(the focal loss paper used) or valid loss normalization

   [CUDA version](focal_loss_v2/sigmoid_focal_loss_v2-inl.h). It supports dymanic grad scale, i.e., grad_scale is a variable input to the operator instead of a fixed parameter. This op can be used in Cascade RetinaNet.

2. **Smooth_l1_loss v2**
   
   [NDArray version](smooth_l1_v2/smooth_l1_v2.py). The difference between the normal smooth_l1 loss is that, its sigma is a variable to pass into the op.

3. **Wing loss**
   
   [NDArray version](wing_loss/wing_loss.py). A reproduction of [Wing loss](https://arxiv.org/pdf/1711.06753.pdf) from CVPR2018.
   Wing loss is proposed for facial landmark localisation, but may benefits human keypoints detection as well.

4. **Bounding_box op (with legacy_box param)**

   [CUDA version](bounding_box_op/bounding_box-inl.h). By default, the bounding_box operator in newest MXNet supposes box width(height) = x2(y2)-x1(y1). However, this assumption is not compatible with previous code. For example, the width of box in PASCAL VOC is usually calculated as x2-x1+1. This is common in many codebase, and could hurt performance(~1 AP in my experiments) due to inconsistency. Therefore, I add the `legacy_box` param in bounding_box op for the backward compatibility.
