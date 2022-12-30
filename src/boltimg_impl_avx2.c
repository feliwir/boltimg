#include "boltimg.h"
#include <immintrin.h>

#ifdef __GNUC__
#define AVX2_TGT __attribute__((__target__("avx2")))
#else
#define AVX2_TGT
#endif

#define USHRT_DIVISOR (float)USHRT_MAX

// AVX2
AVX2_TGT static inline void conv_uint16_float32_norm_avx2_step(__m256i short_pack, __m256 *float_pack_low, __m256 *float_pack_high, __m256 scalar)
{
    __m256i int_pack_low = _mm256_unpacklo_epi16(short_pack, _mm256_set1_epi16(0));
    __m256i int_pack_high = _mm256_unpackhi_epi16(short_pack, _mm256_set1_epi16(0));
    *float_pack_low = _mm256_cvtepi32_ps(int_pack_low);
    *float_pack_high = _mm256_cvtepi32_ps(int_pack_high);
    *float_pack_low = _mm256_div_ps(*float_pack_low, scalar);
    *float_pack_high = _mm256_div_ps(*float_pack_high, scalar);
}

AVX2_TGT static int conv_uint16_float32_norm_avx2_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(USHRT_DIVISOR);
    __m256 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m256i short_pack = _mm256_load_si256((__m256i *)&src[idx]);
        conv_uint16_float32_norm_avx2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm256_stream_ps(&dst[idx], float_pack_low);
        _mm256_stream_ps(&dst[idx + 8], float_pack_high);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

AVX2_TGT static int conv_uint16_float32_norm_avx2_unaligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(USHRT_DIVISOR);
    __m256 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m256i short_pack = _mm256_loadu_si256((__m256i *)&src[idx]);
        conv_uint16_float32_norm_avx2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm256_storeu_ps(&dst[idx], float_pack_low);
        _mm256_storeu_ps(&dst[idx + 8], float_pack_high);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint16_float32_norm_avx2(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m256));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m256));

    if (src_aligned && dst_aligned)
        return conv_uint16_float32_norm_avx2_aligned(w, h, c, src, dst);
    else
        return conv_uint16_float32_norm_avx2_unaligned(w, h, c, src, dst);
}