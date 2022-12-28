#include <benchmark/benchmark.h>
#include <new>
#include <boltimg.h>

const size_t WIDTH = 10000;
const size_t HEIGHT = 10000;

static void BM_UInt16Float32Norm_Scalar(benchmark::State &state)
{
  BoltContext ctx;
  bolt_ctx_init(&ctx, BOLT_HL_SCALAR);

  uint16_t *src = new uint16_t[WIDTH * HEIGHT];
  float *dst = new float[WIDTH * HEIGHT];

  for (auto _ : state)
  {
    bolt_conv_uint16_float32_norm(&ctx, WIDTH, HEIGHT, src, dst);
  }

  delete[] src;
  delete[] dst;
}

static void BM_UInt16Float32Norm_AVX(benchmark::State &state)
{
  BoltContext ctx;
  bolt_ctx_init(&ctx, BOLT_HL_AVX);

  uint16_t *src = new (std::align_val_t(32)) uint16_t[WIDTH * HEIGHT];
  float *dst = new (std::align_val_t(32)) float[WIDTH * HEIGHT];

  for (auto _ : state)
  {
    bolt_conv_uint16_float32_norm(&ctx, WIDTH, HEIGHT, src, dst);
  }

  delete[] src;
  delete[] dst;
}

// Register the function as a benchmark
BENCHMARK(BM_UInt16Float32Norm_Scalar);
BENCHMARK(BM_UInt16Float32Norm_AVX);

// Run the benchmark
BENCHMARK_MAIN();