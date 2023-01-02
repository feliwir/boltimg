#include "boltimg.h"

// TODO: we can remove this no-sse later. But it's useful as a reference
#ifdef __GNUC__
#define SCALAR_TGT __attribute__((__target__("no-sse")))
#else
#define SCALAR_TGT
#endif

#define UCHAR_DIVISOR (float)UCHAR_MAX
#define USHRT_DIVISOR (float)USHRT_MAX

int conv_uint8_float32_norm_scalar(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
  for (size_t idx = 0; idx < h * w * c; ++idx)
  {
    dst[idx] = src[idx] / UCHAR_DIVISOR;
  }
  return BOLT_ERR_SUCCESS;
}

int conv_uint16_float32_norm_scalar(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
  for (size_t idx = 0; idx < h * w * c; ++idx)
  {
    dst[idx] = src[idx] / USHRT_DIVISOR;
  }
  return BOLT_ERR_SUCCESS;
}