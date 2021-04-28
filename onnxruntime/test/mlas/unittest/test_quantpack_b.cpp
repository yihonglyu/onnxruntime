// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

template <bool BIsSigned>
class MlasQuantizePackBTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<uint8_t> BufferOutput;
  MatrixGuardBuffer<uint8_t> BufferOutputReference;

  void GenerateReference(const float* Input, uint8_t* OutputReference, size_t N, size_t K, float Scale, uint8_t ZeroPoint) {
    if (BIsSigned) {
      std::vector<int8_t> intermediate(N * K);
      MlasQuantizeLinear<int8_t>(Input, intermediate.data(), N * K, Scale, ZeroPoint);
      MlasGemmPackB(N, K, (uint8_t*)intermediate.data(), N, BIsSigned, OutputReference, nullptr);
    } else {
      std::vector<uint8_t> intermediate(N * K);
      MlasQuantizeLinear<uint8_t>(Input, intermediate.data(), N * K, Scale, ZeroPoint);
      MlasGemmPackB(N, K, intermediate.data(), N, BIsSigned, OutputReference, nullptr);
    }
  }

  void TestUnitBlock() {

    size_t N = 16;
    size_t K = 4;

    float* Input = BufferInput.GetBuffer(N * K);
    size_t packed_b_size = MlasGemmPackBSize(N, K, BIsSigned);
    uint8_t* Output = BufferOutput.GetBuffer(packed_b_size);
    uint8_t* OutputReference = BufferOutputReference.GetBuffer(packed_b_size);
    std::fill_n(Output, packed_b_size, uint8_t(10));
    std::fill_n(OutputReference, packed_b_size, uint8_t(10));

    float Scale = 1.f;
    uint8_t ZeroPoint = 0;

    Input[0] = float(0x10);
    for (int i = 1; i < 64; i++) {
      Input[i] = Input[i - 1] + 1;
    }

    GenerateReference(Input, OutputReference, N, K, Scale, ZeroPoint);

    // TODO Add multi-thread tests
    MlasQuantizeLinearPackB(N, K, Input, N, BIsSigned, Scale, ZeroPoint, Output, nullptr);

    ASSERT_EQ(0, memcmp(Output, OutputReference, packed_b_size));
    // TODO need a stand alone pack A function to test values,
    // currently just test memory out of bound errors
  }

  void Test(size_t N, size_t K) {
    float* Input = BufferInput.GetBuffer(N * K);
    size_t packed_b_size = MlasGemmPackBSize(N, K, BIsSigned);
    uint8_t* Output = BufferOutput.GetBuffer(packed_b_size);
    uint8_t* OutputReference = BufferOutputReference.GetBuffer(packed_b_size);
    std::fill_n(Output, packed_b_size, uint8_t(0x55));
    std::fill_n(OutputReference, packed_b_size, uint8_t(0x55));

    std::default_random_engine generator(static_cast<unsigned>(N * K));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 512.f;

    std::uniform_int_distribution<int32_t> zp_distribution(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    uint8_t ZeroPoint = static_cast<uint8_t>(zp_distribution(generator));

    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);
    for (size_t n = 0; n < (N * K); n++) {
      Input[n] = distribution(generator);
    }

    GenerateReference(Input, OutputReference, N, K, Scale, ZeroPoint);

    // TODO Add multi-thread tests
    MlasQuantizeLinearPackB(N, K, Input, N, BIsSigned, Scale, ZeroPoint, Output, nullptr);

    ASSERT_EQ(0, memcmp(Output, OutputReference, packed_b_size));
    // TODO need a stand alone pack A function to test values,
    // currently just test memory out of bound errors
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name(BIsSigned ? "QuantizePackBSignedB" : "QuantizePackBUnsignedB");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    TestUnitBlock();
    Test(1, 15);
    Test(15, 1);
    Test(1, 150);
    Test(150, 1);
    Test(3, 150);
    Test(150, 3);
    for (size_t n = 10; n <= 512; n++) {
      Test(n, 4000 / n);
    }
    Test(1000, 3000);
    Test(2048, 1024);
  }

};

template <>
MlasQuantizePackBTest<true>* MlasTestFixture<MlasQuantizePackBTest<true>>::mlas_tester(nullptr);
template <>
MlasQuantizePackBTest<false>* MlasTestFixture<MlasQuantizePackBTest<false>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasQuantizePackBTest<true>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizePackBTest<false>>::RegisterShortExecute();
  }
  return count;
});
