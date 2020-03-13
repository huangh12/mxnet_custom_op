#include "./sigmoid_focal_loss_v2-inl.h"


namespace mshadow {

template<typename Dtype>
inline void SigmoidFocalLossV2Forward(const Tensor<cpu, 3, Dtype> &out_prob,
                                    const Tensor<cpu, 3, Dtype> &out_loss,
                                    const Tensor<cpu, 3, Dtype> &in_data,
                                    const Tensor<cpu, 2, Dtype> &in_label,
                                    const Tensor<cpu, 2, Dtype> &in_label_weight,
                                    const float ignore_label,
                                    const float nonfocal_label,
                                    const float alpha,
                                    const float gamma) {
  return;
}

template<typename Dtype>
inline void SigmoidFocalLossV2Backward(const Tensor<cpu, 3, Dtype> &in_data_grad,
                                     const Tensor<cpu, 3, Dtype> &in_data,
                                     const Tensor<cpu, 2, Dtype> &in_label,
                                     const Tensor<cpu, 2, Dtype> &in_label_weight,
                                     const float ignore_label,
                                     const float nonfocal_label,
                                     const float alpha,
                                     const float gamma) {
  return;  
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(SigmoidFocalLossV2Param param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SigmoidFocalLossV2Op<cpu, DType>(param);
  });
  return op;
}

Operator *SigmoidFocalLossV2Prop::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(SigmoidFocalLossV2Param);

MXNET_REGISTER_OP_PROPERTY(SigmoidFocalLossV2, SigmoidFocalLossV2Prop)
.describe(R"code(SigmoidFocalLossV2)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(SigmoidFocalLossV2Param::__FIELDS__());
}  // namespace op
}  // namespace mxnet
