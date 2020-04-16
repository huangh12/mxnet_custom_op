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

5. **TopK op (with do_sort param)**

   [CUDA version](topk/ordering_op-inl.h). The influence of `do_sort` param is as below. When `do_sort` is True(by default),
   it behaves exactly same as the official topk operator in MXNet. 
   
   ```Python
   >>> a

   [[0.08712929 0.6481719  0.0202184  0.36824155 0.83261985]
   [0.95715517 0.77815676 0.14035077 0.87001216 0.87008727]
   [0.9786183  0.47360805 0.7991586  0.8009108  0.46147937]]
   <NDArray 3x5 @cpu(0)>
   >>> mx.nd.topk(a, k=3, axis=-1, do_sort=1, ret_typ="value")

   [[0.83261985 0.6481719  0.36824155]
   [0.95715517 0.87008727 0.87001216]
   [0.9786183  0.8009108  0.7991586 ]]
   <NDArray 3x3 @cpu(0)>
   >>> mx.nd.topk(a, k=3, axis=-1, do_sort=0, ret_typ="value")

   [[0.6481719  0.36824155 0.83261985]
   [0.95715517 0.87001216 0.87008727]
   [0.9786183  0.7991586  0.8009108 ]]
   <NDArray 3x3 @cpu(0)>

   ```

   The benefits of having `do_sort` param is that, we can construct a dynamic indexing scheme by using topk op in `do_sort=0` mode. The following example shows dynamically idexing a tensor(a) with another tensor(b), and the output(c) will change along with b. Note this hasn't been supported by official MXNet, but can be implemented with this modified topk operator.

   ```Python
   def arr_pick(self, F, arr, indicate, num):
      ## NOTE: below adding 1 is more than necessary!!
      max_gap_value = F.max(arr) - F.min(arr) + 1  # (1,)
      max_gap_value = F.reshape(max_gap_value, (1, 1, 1))
      max_gap_arr = F.broadcast_mul(indicate, max_gap_value)
      arr = F.broadcast_plus(arr, max_gap_arr)     # (batch_size, mask_dim, h*w*priors)
      arr = F.topk(arr, k=num, axis=-1, do_sort=0, ret_typ="value")
      res = F.broadcast_minus(arr, max_gap_value)      # (batch_size, mask_dim, mask_num)
      return res   

   a = mx.nd.arange(60, ctx=mx.gpu()).reshape((1,6,10))
   b = mx.nd.array([[[0,1,0,1,0,1,0,1,0,1]]], ctx=mx.gpu())
   c = arr_pick(mx.nd, a, b, 5)
   print(a)
   print(c)
   
   # output
   [[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
   [10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]
   [20. 21. 22. 23. 24. 25. 26. 27. 28. 29.]
   [30. 31. 32. 33. 34. 35. 36. 37. 38. 39.]
   [40. 41. 42. 43. 44. 45. 46. 47. 48. 49.]
   [50. 51. 52. 53. 54. 55. 56. 57. 58. 59.]]]
   <NDArray 1x6x10 @gpu(0)>

   [[[ 1.  3.  5.  7.  9.]
   [11. 13. 15. 17. 19.]
   [21. 23. 25. 27. 29.]
   [31. 33. 35. 37. 39.]
   [41. 43. 45. 47. 49.]
   [51. 53. 55. 57. 59.]]]
   <NDArray 1x6x5 @gpu(0)>
   ```

6. **OHEM Sampler**

   [Python version](ohem_sampler/ohem_sampler.py). OHEM sampler is for online hard example mining. See this [paper](https://arxiv.org/abs/1604.03540) for detailed introduction.
   In my implementation, it's realized by adjusting the sample weight before the loss layer, i.e., setting the hard(easy) sample weight to 1(0). Therefore, the inputs to this operator are `conf_data`, `conf_target`, `conf_weight`, and the output is the `new_conf_weight`.