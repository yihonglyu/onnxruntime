// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

template<bool BIsSigned>
class MlasQuantizePackATest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<uint8_t> BufferOutput;
  MatrixGuardBuffer<uint8_t> BufferOutputReference;

  void GenerateReference(const float* Input, uint8_t* OutputReference, size_t M, size_t K, float Scale, uint8_t ZeroPoint) {
    std::vector<uint8_t> intermediate(M * K);
    MlasQuantizeLinear(Input, intermediate.data(), M * K, Scale, ZeroPoint);
    MlasGemmPackA(M, K, BIsSigned, intermediate.data(), K, OutputReference, nullptr);
  }

  void Test(size_t M, size_t K) {
    float* Input = BufferInput.GetBuffer(M * K);
    size_t packed_a_size = MlasGemmPackASize(M, K, BIsSigned);
    uint8_t* Output = BufferOutput.GetBuffer(packed_a_size);
    uint8_t* OutputReference = BufferOutputReference.GetBuffer(packed_a_size);
    std::fill_n(Output, packed_a_size, uint8_t(0));
    std::fill_n(OutputReference, packed_a_size, uint8_t(0));

    std::default_random_engine generator(static_cast<unsigned>(M * K));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 512.f;

    std::uniform_int_distribution<int32_t> zp_distribution(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    uint8_t ZeroPoint = static_cast<uint8_t>(zp_distribution(generator));

    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);
    for (size_t n = 0; n < (M*K); n++) {
      Input[n] = distribution(generator);
    }

    GenerateReference(Input, OutputReference, M, K, Scale, ZeroPoint);

    // TODO Add multi-thread tests
    MlasQuantizeLinearPackA(M, K, Input, K, BIsSigned, Scale, ZeroPoint, Output, nullptr);

    ASSERT_EQ(0, memcmp(Output, OutputReference, packed_a_size));
    // TODO need a stand alone pack A function to test values,
    // currently just test memory out of bound errors
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name(BIsSigned ? "QuantizePackASignedB" : "QuantizePackAUnsignedB");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
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
MlasQuantizePackATest<true>* MlasTestFixture<MlasQuantizePackATest<true>>::mlas_tester(nullptr);
template <>
MlasQuantizePackATest<false>* MlasTestFixture<MlasQuantizePackATest<false>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasQuantizePackATest<true>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizePackATest<false>>::RegisterShortExecute();
  }
  return count;
});
