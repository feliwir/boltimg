#include "boltimg.h"

// TODO: we can remove this no-sse later. But it's useful as a reference
__attribute__((__target__("no-sse"))) int conv_uint16_float32_norm_scalar(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
  for (size_t idx = 0; idx < h * w; ++idx)
  {
    dst[idx] = src[idx] / 65535.0f;
  }
  return BOLT_ERR_SUCCESS;
}