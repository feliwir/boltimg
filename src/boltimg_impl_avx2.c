#include "boltimg.h"
#include <immintrin.h>

// AVX2
 __attribute__((__target__("avx2")))
int conv_uint16_float32_norm_avx(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    size_t idx = 0;
    const __m256 scalar = _mm256_set1_ps(65535.0f);
    for (; idx < (h * w - 16); idx += 16)
    {
        __m256i short_pack = _mm256_load_si256((__m256i *)&src[idx]);
        __m256i int_pack_low = _mm256_unpacklo_epi16(short_pack, _mm256_set1_epi16(0));
        __m256i int_pack_high = _mm256_unpackhi_epi16(short_pack, _mm256_set1_epi16(0));
        __m256 float_pack_low = _mm256_cvtepi32_ps(int_pack_low);
        __m256 float_pack_high = _mm256_cvtepi32_ps(int_pack_high);
        float_pack_low = _mm256_div_ps(float_pack_low, scalar);
        float_pack_high = _mm256_div_ps(float_pack_high, scalar);
        _mm256_store_ps(&dst[idx], float_pack_low);
        _mm256_store_ps(&dst[idx + 8], float_pack_high);
    }
    for (; idx < h * w; ++idx)
    {
        dst[idx] = src[idx] / 65535.0f;
    }

    return BOLT_ERR_SUCCESS;
}