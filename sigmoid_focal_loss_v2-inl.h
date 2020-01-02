#ifndef MXNET_OPERATOR_SIGMOID_FOCAL_LOSS_V2_INL_H_
#define MXNET_OPERATOR_SIGMOID_FOCAL_LOSS_V2_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"

#define FOCAL_LOSS_V2_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace SigmoidFocalLossV2 {
enum SigmoidFocalLossV2OpInputs {kData, kLabel, kLabelweight, kGradscale};
enum SigmoidFocalLossV2OpOutputs {kProb, kLoss};
enum SigmoidFocalLossV2OpResource {kTempSpace};
}  // SigmoidFocalLossV2

struct SigmoidFocalLossV2Param : public dmlc::Parameter<SigmoidFocalLossV2Param> {
  // float grad_scale;
  float ignore_label;
  float nonfocal_label;
  float alpha;
  float gamma;
  DMLC_DECLARE_PARAMETER(SigmoidFocalLossV2Param) {
    // DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    // .describe("Scales the gradient by a float factor.");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("The instances whose `labels` == `ignore_label` will be ignored "
              "during backward, if `use_ignore` is set to ``true``).");  
    DMLC_DECLARE_FIELD(nonfocal_label).set_default(-1.0f)
    .describe("The instances whose `labels` == `ignore_label` will be ignored "
              "during backward, if `use_ignore` is set to ``true``).");                
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f)
    .describe("Focal Loss's alpha hyper-parameter.");
    DMLC_DECLARE_FIELD(gamma).set_default(1.0f)
    .describe("Focal Loss's gamma hyper-parameter.");    
  }
};

template<typename xpu, typename DType>
class SigmoidFocalLossV2Op : public Operator {
 public:
  explicit SigmoidFocalLossV2Op(SigmoidFocalLossV2Param p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 4);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[SigmoidFocalLossV2::kData].size(0);
    int k = in_data[SigmoidFocalLossV2::kData].size(1);


    Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[SigmoidFocalLossV2::kData].Size()/n/k));
    Shape<2> s2 = Shape2(s3[0], s3[2]);

    Tensor<xpu, 3, DType> data = in_data[SigmoidFocalLossV2::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 2, DType> label = in_data[SigmoidFocalLossV2::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 2, DType> label_weight = in_data[SigmoidFocalLossV2::kLabelweight].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 3, DType> out_prob = out_data[SigmoidFocalLossV2::kProb].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out_loss = out_data[SigmoidFocalLossV2::kLoss].get_with_shape<xpu, 3, DType>(s3, s);
    
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(label_weight.CheckContiguous(), true);
    CHECK_EQ(out_prob.CheckContiguous(), true);
    CHECK_EQ(out_loss.CheckContiguous(), true);

    SigmoidFocalLossV2Forward(out_prob, out_loss, data, label, label_weight, param_.ignore_label, param_.nonfocal_label, param_.alpha, param_.gamma);

    int num_pos_label = 0;
    Tensor<cpu, 2, DType> workspace = ctx.requested[SigmoidFocalLossV2::kTempSpace].get_host_space_typed<2, DType>(label.shape_);
    Copy(workspace, label, label.stream_);
    for (index_t i = 0; i < workspace.size(0); ++i) {
      for (index_t j = 0; j < workspace.size(1); ++j) {
        if (static_cast<int>(workspace[i][j]) > 0) {
          ++num_pos_label;
        }
      }
    }
    num_pos_label = num_pos_label == 0 ? 1 : num_pos_label;
    out_loss *= DType(1.0 / num_pos_label);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 4U);
    CHECK_GE(in_grad.size(), 1U);
    CHECK_GE(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[SigmoidFocalLossV2::kData].size(0);
    int k = in_data[SigmoidFocalLossV2::kData].size(1);
    Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[SigmoidFocalLossV2::kData].Size()/n/k));
    Shape<2> s2 = Shape2(s3[0], s3[2]);
    Shape<2> s4 = Shape2(s3[0], 1);

    Tensor<xpu, 3, DType> data = in_data[SigmoidFocalLossV2::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 2, DType> label = in_data[SigmoidFocalLossV2::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 2, DType> label_weight = in_data[SigmoidFocalLossV2::kLabelweight].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 2, DType> grad_scale = in_data[SigmoidFocalLossV2::kGradscale].get_with_shape<xpu, 2, DType>(s4, s);
    Tensor<xpu, 3, DType> in_data_grad = in_grad[SigmoidFocalLossV2::kData].get_with_shape<xpu, 3, DType>(s3, s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(in_data_grad.CheckContiguous(), true);

    if (kAddTo == req[SigmoidFocalLossV2::kData] || kWriteTo == req[SigmoidFocalLossV2::kData]) {
      if (kWriteTo == req[SigmoidFocalLossV2::kData]) {
        in_data_grad = 0.0f;
      }

      SigmoidFocalLossV2Backward(in_data_grad, data, label, label_weight, param_.ignore_label, param_.nonfocal_label, param_.alpha, param_.gamma);

      int num_pos_label = 0;
      Tensor<cpu, 2, DType> workspace = ctx.requested[SigmoidFocalLossV2::kTempSpace].get_host_space_typed<2, DType>(label.shape_);
      Copy(workspace, label, label.stream_);
      for (index_t i = 0; i < workspace.size(0); ++i) {
        for (index_t j = 0; j < workspace.size(1); ++j) {
          if (static_cast<int>(workspace[i][j]) > 0) {
            ++num_pos_label;
          }
        }
      }
      num_pos_label = num_pos_label == 0 ? 1 : num_pos_label;
      // in_data_grad *= DType(param_.grad_scale / num_pos_label);


      std::vector<float> cpu_grad_scale(n);
      FOCAL_LOSS_V2_CUDA_CHECK(cudaMemcpy(&cpu_grad_scale[0], grad_scale.dptr_,
                                   sizeof(float) * cpu_grad_scale.size(),
                                   cudaMemcpyDeviceToHost));
      in_data_grad *= DType(cpu_grad_scale[0] / num_pos_label);
    }
    
    if (kWriteTo == req[SigmoidFocalLossV2::kLabel]) {
      Tensor<xpu, 2, DType> in_label_grad = in_grad[SigmoidFocalLossV2::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
      CHECK_EQ(in_label_grad.CheckContiguous(), true);
      in_label_grad = 0.0f;
    }
    if (kWriteTo == req[SigmoidFocalLossV2::kLabelweight]) {
      Tensor<xpu, 2, DType> in_labelweight_grad = in_grad[SigmoidFocalLossV2::kLabelweight].get_with_shape<xpu, 2, DType>(s2, s);
      CHECK_EQ(in_labelweight_grad.CheckContiguous(), true);
      in_labelweight_grad = 0.0f;
    }    
    if (kWriteTo == req[SigmoidFocalLossV2::kGradscale]) {
      Tensor<xpu, 2, DType> in_gradscale_grad = in_grad[SigmoidFocalLossV2::kGradscale].get_with_shape<xpu, 2, DType>(s4, s);
      CHECK_EQ(in_gradscale_grad.CheckContiguous(), true);
      in_gradscale_grad = 0.0f;
    }

  }

 private:
  SigmoidFocalLossV2Param param_;
};  // class SigmoidFocalLossV2Op

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SigmoidFocalLossV2Param param, int dtype);

#if DMLC_USE_CXX11
class SigmoidFocalLossV2Prop : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label", "label_weight", "grad_scale"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output_prob", "output_loss"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 4U) << "Input:[data, label, label_weight, grad_scale]";
    TShape &dshape = in_shape->at(SigmoidFocalLossV2::kData);
    if (dshape.ndim() == 0) return false;

    // for kLabel
    TShape lshape1 = Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]);
    TShape lshape2(dshape.ndim() - 1);
    lshape2[0] = dshape[0];
    for (index_t i = 2; i < dshape.ndim(); ++i)
      lshape2[i-1] = dshape[i];
    TShape lshape3 = dshape;
    lshape3[1] = 1;
    if (in_shape->at(SigmoidFocalLossV2::kLabel).ndim() == 0) {
      in_shape->at(SigmoidFocalLossV2::kLabel) = lshape1;
    } else if (in_shape->at(SigmoidFocalLossV2::kLabel) == lshape1) {
    } else if (in_shape->at(SigmoidFocalLossV2::kLabel) == lshape2) {
    } else if (in_shape->at(SigmoidFocalLossV2::kLabel) == lshape3) {
    } else {
      std::ostringstream os;
      os << "Expecting " << lshape1 << " or " << lshape2
         << ". But got " << in_shape->at(SigmoidFocalLossV2::kLabel);
      throw InferShapeError(os.str(), SigmoidFocalLossV2::kLabel);
    }

    // for kLabelweight 
    if (in_shape->at(SigmoidFocalLossV2::kLabelweight).ndim() == 0) {
      in_shape->at(SigmoidFocalLossV2::kLabelweight) = in_shape->at(SigmoidFocalLossV2::kLabel);
    } else if (in_shape->at(SigmoidFocalLossV2::kLabel) == in_shape->at(SigmoidFocalLossV2::kLabel)) {
    } else {
      std::ostringstream os;
      os << "Expecting kLabelweight shape same as kLabel " << in_shape->at(SigmoidFocalLossV2::kLabel)
         << ". But got " << in_shape->at(SigmoidFocalLossV2::kLabelweight);
      throw InferShapeError(os.str(), SigmoidFocalLossV2::kLabelweight);
    }

    // for kGradscale 
    TShape gshape = Shape2(dshape[0],1);
    if (in_shape->at(SigmoidFocalLossV2::kGradscale).ndim() == 0) {
      in_shape->at(SigmoidFocalLossV2::kGradscale) = gshape;
    } else if (in_shape->at(SigmoidFocalLossV2::kGradscale) == gshape) {
    } else {
      std::ostringstream os;
      os << "Expecting kGradscale shape should be " << gshape
         << ". But got " << in_shape->at(SigmoidFocalLossV2::kGradscale);
      throw InferShapeError(os.str(), SigmoidFocalLossV2::kGradscale);
    }

    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    SigmoidFocalLossV2Prop* sigmoid_focal_loss_sym = new SigmoidFocalLossV2Prop();
    sigmoid_focal_loss_sym->param_ = this->param_;
    return sigmoid_focal_loss_sym;
  }

  std::string TypeString() const override {
    return "SigmoidFocalLossV2";
  }

  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[SigmoidFocalLossV2::kData], in_data[SigmoidFocalLossV2::kLabel], in_data[SigmoidFocalLossV2::kLabelweight], 
    in_data[SigmoidFocalLossV2::kGradscale]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SigmoidFocalLossV2Param param_;
};  // class SigmoidFocalLossV2Prop
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SIGMOID_FOCAL_LOSS_V2_INL_H_
