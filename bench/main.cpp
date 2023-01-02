#include <benchmark/benchmark.h>
#include <new>
#include <boltimg.h>

constexpr size_t W = 10000;
constexpr size_t H = 10000;

static void BM_UInt8Float32Norm(benchmark::State &state)
{
  BoltContext ctx;
  BoltHardwareLevel hl = static_cast<BoltHardwareLevel>(state.range(0));
  bool aligned = state.range(1);
  if (bolt_ctx_init(&ctx, hl) != BOLT_ERR_SUCCESS)
  {
    state.SkipWithError("Not supported");
    return;
  }

  uint8_t *src = nullptr;
  float *dst = nullptr;

  if (aligned)
  {
    src = new (std::align_val_t(32)) uint8_t[W * H];
    dst = new (std::align_val_t(32)) float[W * H];
  }
  else
  {
    // Make sure this is unaligned
    src = new uint8_t[W * H + 1];
    dst = new float[W * H + 1];
    src += 1;
    dst += 1;
  }

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(bolt_conv_uint8_float32_norm(&ctx, W, H, 1, src, dst));
  }

  if (!aligned)
  {
    src -= 1;
    dst -= 1;
  }

  delete[] src;
  delete[] dst;
}

static void BM_UInt16Float32Norm(benchmark::State &state)
{
  BoltContext ctx;
  BoltHardwareLevel hl = static_cast<BoltHardwareLevel>(state.range(0));
  bool aligned = state.range(1);
  if (bolt_ctx_init(&ctx, hl) != BOLT_ERR_SUCCESS)
  {
    state.SkipWithError("Not supported");
    return;
  }

  uint16_t *src = nullptr;
  float *dst = nullptr;

  if (aligned)
  {
    src = new (std::align_val_t(32)) uint16_t[W * H];
    dst = new (std::align_val_t(32)) float[W * H];
  }
  else
  {
    // Make sure this is unaligned
    src = new uint16_t[W * H + 1];
    dst = new float[W * H + 1];
    src += 1;
    dst += 1;
  }

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(bolt_conv_uint16_float32_norm(&ctx, W, H, 1, src, dst));
  }

  if (!aligned)
  {
    src -= 1;
    dst -= 1;
  }

  delete[] src;
  delete[] dst;
}

// Register the function as a benchmark
// clang-format off
BENCHMARK(BM_UInt8Float32Norm)->Args({BOLT_HL_AVX512, true})
                              ->Args({BOLT_HL_AVX512, false})
                              ->Args({BOLT_HL_AVX2, true})
                              ->Args({BOLT_HL_AVX2, false})
                              ->Args({BOLT_HL_SSE4, true})
                              ->Args({BOLT_HL_SSE4, false})
                              ->Args({BOLT_HL_SCALAR, true})
                              ->Args({BOLT_HL_SCALAR, false});

BENCHMARK(BM_UInt16Float32Norm) ->Args({BOLT_HL_AVX512, true})
                                ->Args({BOLT_HL_AVX512, false})
                                ->Args({BOLT_HL_AVX2, true})
                                ->Args({BOLT_HL_AVX2, false})
                                ->Args({BOLT_HL_SSE4, true})
                                ->Args({BOLT_HL_SSE4, false})
                                ->Args({BOLT_HL_SCALAR, true})
                                ->Args({BOLT_HL_SCALAR, false});
// clang-format on

// Run the benchmark
BENCHMARK_MAIN();