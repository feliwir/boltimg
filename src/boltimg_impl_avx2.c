#include "boltimg.h"
#include <immintrin.h>

#ifdef __GNUC__
#define AVX2_TGT __attribute__((__target__("avx2")))
#else
#define AVX2_TGT
#endif

// UINT8
#define UCHAR_DIVISOR (float)UCHAR_MAX

AVX2_TGT static inline void conv_uint8_float32_norm_avx2_step(__m128i byte_pack, __m256 *float_pack_lo, __m256 *float_pack_hi, __m256 scalar)
{
    __m256i short_pack = _mm256_cvtepu8_epi16(byte_pack);
    __m256i int_pack_lo = _mm256_unpacklo_epi16(short_pack, _mm256_set1_epi16(0));
    __m256i int_pack_hi = _mm256_unpackhi_epi16(short_pack, _mm256_set1_epi16(0));

    *float_pack_lo = _mm256_cvtepi32_ps(int_pack_lo);
    *float_pack_hi = _mm256_cvtepi32_ps(int_pack_hi);

    *float_pack_lo = _mm256_div_ps(*float_pack_lo, scalar);
    *float_pack_hi = _mm256_div_ps(*float_pack_hi, scalar);
}

AVX2_TGT static int conv_uint8_float32_norm_avx2_aligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(UCHAR_DIVISOR);
    __m256 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_uint8_float32_norm_avx2_step(byte_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm256_stream_ps(&dst[idx], float_pack_lo);
        _mm256_stream_ps(&dst[idx + 8], float_pack_hi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

AVX2_TGT static int conv_uint8_float32_norm_avx2_unaligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(UCHAR_DIVISOR);
    __m256 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_uint8_float32_norm_avx2_step(byte_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm256_storeu_ps(&dst[idx], float_pack_lo);
        _mm256_storeu_ps(&dst[idx + 8], float_pack_hi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint8_float32_norm_avx2(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m256));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m256));

    if (src_aligned && dst_aligned)
        return conv_uint8_float32_norm_avx2_aligned(w, h, c, src, dst);
    else
        return conv_uint8_float32_norm_avx2_unaligned(w, h, c, src, dst);
}

// UINT16
#define USHRT_DIVISOR (float)USHRT_MAX

AVX2_TGT static inline void conv_uint16_float32_norm_avx2_step(__m256i short_pack, __m256 *float_pack_lo, __m256 *float_pack_hi, __m256 scalar)
{
    __m256i int_pack_lo = _mm256_unpacklo_epi16(short_pack, _mm256_set1_epi16(0));
    __m256i int_pack_hi = _mm256_unpackhi_epi16(short_pack, _mm256_set1_epi16(0));
    *float_pack_lo = _mm256_cvtepi32_ps(int_pack_lo);
    *float_pack_hi = _mm256_cvtepi32_ps(int_pack_hi);
    *float_pack_lo = _mm256_div_ps(*float_pack_lo, scalar);
    *float_pack_hi = _mm256_div_ps(*float_pack_hi, scalar);
}

AVX2_TGT static int conv_uint16_float32_norm_avx2_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(USHRT_DIVISOR);
    __m256 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m256i short_pack = _mm256_load_si256((__m256i *)&src[idx]);
        conv_uint16_float32_norm_avx2_step(short_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm256_stream_ps(&dst[idx], float_pack_lo);
        _mm256_stream_ps(&dst[idx + 8], float_pack_hi);
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
    __m256 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m256i short_pack = _mm256_loadu_si256((__m256i *)&src[idx]);
        conv_uint16_float32_norm_avx2_step(short_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm256_storeu_ps(&dst[idx], float_pack_lo);
        _mm256_storeu_ps(&dst[idx + 8], float_pack_hi);
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