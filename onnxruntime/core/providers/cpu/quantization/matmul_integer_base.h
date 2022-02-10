// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/quantization/quantization.h"

namespace onnxruntime {

class MatMulIntegerBase : public OpKernel {
 public:
  MatMulIntegerBase(const OpKernelInfo& info) : OpKernel(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override {
    is_packed = false;

    // only pack Matrix B
    if (input_idx == GetBIdx()) {
      // Only handle the common case of a 2D weight matrix. Additional matrices
      // could be handled by stacking the packed buffers.
      b_shape_ = tensor.Shape();
      if (b_shape_.NumDimensions() != 2) {
        return Status::OK();
      }

      auto a_elem_type = Node().InputDefs()[GetAIdx()]->TypeAsProto()->tensor_type().elem_type();
      bool a_is_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == a_elem_type;

      b_is_signed_ = tensor.IsDataType<int8_t>();

      size_t K = static_cast<size_t>(b_shape_[0]);
      size_t N = static_cast<size_t>(b_shape_[1]);

      const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());

      BufferUniquePtr b_trans_buffer;
      if (IsBTransposed()) {
        std::swap(K, N);
        b_data = quantization::TransPoseInputData(b_data, b_trans_buffer, alloc, N, K);
      }

      if (TrySymmQuantPrepack(alloc, (const int8_t*)(b_data), N, K, a_is_signed)) {
        is_packed = true;
        return Status::OK();
      }

      const size_t packed_b_size = MlasGemmPackBSize(N, K, a_is_signed, b_is_signed_);
      if (packed_b_size == 0) {
        return Status::OK();
      }

      auto* packed_b_data = alloc->Alloc(packed_b_size);

      // Initialize memory to 0 as there could be some padding associated with pre-packed
      // buffer memory and we don not want it uninitialized and generate different hashes
      // if and when we try to cache this pre-packed buffer for sharing between sessions.
      memset(packed_b_data, 0, packed_b_size);

      packed_b_ = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
      MlasGemmPackB(N, K, b_data, N, a_is_signed, b_is_signed_, packed_b_data);

      bool share_prepacked_weights = (prepacked_weights != nullptr);
      if (share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_b_));
        prepacked_weights->buffer_sizes_.push_back(packed_b_size);
      }

      is_packed = true;
    }
    return Status::OK();
  }

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override {
    used_shared_buffers = false;

    if (input_idx == GetBIdx()) {
      used_shared_buffers = true;
      packed_b_ = std::move(prepacked_buffers[0]);
    }

    return Status::OK();
  }

 protected:
  /**
   * @return input index of Matrix B, the weight tensor 
  */
  virtual int GetAIdx() const { return 0; }
  virtual int GetBIdx() const = 0;

  virtual bool IsBTransposed() const {
    return false;
  }

  // Check if quantization parameter of B is supported.
  // It should be in one of the formats below:
  // 1. Scalar
  // 2. 1D tensor with size equal to 1 or last dimension of B_shape if B_shape is a 2D tensor
  // 3. Equal to B_shape except that the second to last is 1
  bool IsBQuantParamSupported(const TensorShape& B_quant_param_shape, const TensorShape& B_shape) const {
    int64_t B_quant_param_rank = B_quant_param_shape.NumDimensions();
    int64_t B_shape_rank = B_shape.NumDimensions();
    if (B_quant_param_rank == 0 ||                                       //scalar
        (B_quant_param_rank == 1 && B_quant_param_shape.Size() == 1)) {  // 1D tensor with size 1
      return true;
    }

    if (B_quant_param_rank == 1 &&
        B_shape_rank == 2 &&
        B_quant_param_shape[B_quant_param_rank - 1] == B_shape[B_shape_rank - 1]) {
      return true;
    }

    if (B_quant_param_rank != B_shape_rank ||
        B_quant_param_rank <= 1 ||
        B_quant_param_shape[B_quant_param_rank - 2] != 1) {
      return false;
    }

    for (int64_t rank = 0; rank < B_quant_param_rank; rank++) {
      if (rank != B_quant_param_rank - 2 &&
          B_quant_param_shape[rank] != B_shape[rank]) {
        return false;
      }
    }

    return true;
  }

  inline const Tensor* TryGetConstInput(int inputIdx) {
    const Tensor* tensor = nullptr;

    if (Info().TryGetConstantInput(inputIdx, &tensor)) {
      return tensor;
    }
    return nullptr;
  }

  /**
   * @brief Get zero point of Activation tensor. Used by PrePack for symmetric
   *        quant packing.  Returns nullptr when symmetric quant gemm not
   *        supported.
  */
  virtual const Tensor* GetAZeroPoint() { return nullptr; }

  /**
   * @brief Get zero point of the weight matrix. Used by PrePack for symmetric
   *        quant packing. Returns nullptr when symmetric quant gemm not
   *        supported.
  */
  virtual const Tensor* GetWZeroPoint() { return nullptr; }

  bool b_is_signed_{true};
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;
  bool symm_packed_{false};

 private:

  /**
   * @brief Attempt symmetric quantized pre-packing of weight tensor.
   * @param alloc         memory allocator
   * @param weight_data   weight tensor data
   * @param N             No. columns
   * @param K             No. rows
   * @param a_is_signed   Whether activation tensor is signed int8 
   * @return              true if symmetric quant prepacking successful
  */
  bool TrySymmQuantPrepack(AllocatorPtr alloc, const int8_t* weight_data,
      size_t N, size_t K, bool a_is_signed) {

    if (!b_is_signed_) {
      // how can it be symmetric if weights are all non-negative?
      return false;
    }

    const Tensor* AZeroPoint = GetAZeroPoint();
    if (nullptr == AZeroPoint || !IsScalarOr1ElementVector(AZeroPoint)) {
      return false;
    }

    const Tensor* WZeroPoint = GetWZeroPoint();
    if (nullptr == WZeroPoint) {
      return false;
    }

    int32_t X_zero_point_value;
    if (a_is_signed) {
      X_zero_point_value = *(AZeroPoint->template Data<int8_t>());
    } else {
      X_zero_point_value = *(AZeroPoint->template Data<uint8_t>());
    }

    const size_t W_zero_point_size = static_cast<size_t>(WZeroPoint->Shape().Size());
    const auto* W_zero_point_data = WZeroPoint->Data<int8_t>();
    if (!std::all_of(W_zero_point_data, W_zero_point_data + W_zero_point_size, [](int8_t v) { return v == 0; })) {
      // Symmetric means weight zero point must be zero
      return false;
    }

    size_t pack_size = MlasSymmQgemmPackBSize(N, K, a_is_signed);
    if (pack_size == 0) {
      return false;
    }

    auto* packed_b_data = alloc->Alloc(pack_size);

    // TODO!! What about sharing prepacked weight? do we need to consider them?

    packed_b_ = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
    MlasSymmQgemmPackB(N, K, weight_data, N, a_is_signed, X_zero_point_value, packed_b_data);
    symm_packed_ = true;
    return true;
  }
};

}  // namespace onnxruntime
