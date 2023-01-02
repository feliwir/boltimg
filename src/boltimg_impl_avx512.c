#include "boltimg.h"
#include <immintrin.h>

#ifdef __GNUC__
#define AVX512_TGT __attribute__((__target__("avx512bw")))
#else
#define AVX512_TGT
#endif

// UINT8
#define UCHAR_DIVISOR (float)UCHAR_MAX

AVX512_TGT static inline void conv_uint8_float32_norm_avx512_step(__m256i byte_pack, __m512 *float_pack_lo, __m512 *float_pack_hi, __m512 scalar)
{
    __m512i short_pack = _mm512_cvtepu8_epi16(byte_pack);
    __m512i int_pack_lo = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    __m512i int_pack_hi = _mm512_unpackhi_epi16(short_pack, _mm512_set1_epi16(0));

    *float_pack_lo = _mm512_cvtepi32_ps(int_pack_lo);
    *float_pack_hi = _mm512_cvtepi32_ps(int_pack_hi);

    *float_pack_lo = _mm512_div_ps(*float_pack_lo, scalar);
    *float_pack_hi = _mm512_div_ps(*float_pack_hi, scalar);
}

AVX512_TGT static int conv_uint8_float32_norm_avx512_aligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(UCHAR_DIVISOR);
    __m512 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 32); idx += 32)
    {
        __m256i byte_pack = _mm256_load_si256((__m256i *)&src[idx]);
        conv_uint8_float32_norm_avx512_step(byte_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm512_stream_ps(&dst[idx], float_pack_lo);
        _mm512_stream_ps(&dst[idx + 16], float_pack_hi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

AVX512_TGT static int conv_uint8_float32_norm_avx512_unaligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(UCHAR_DIVISOR);
    __m512 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m256i byte_pack = _mm256_loadu_si256((__m256i *)&src[idx]);
        conv_uint8_float32_norm_avx512_step(byte_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm512_storeu_ps(&dst[idx], float_pack_lo);
        _mm512_storeu_ps(&dst[idx + 8], float_pack_hi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint8_float32_norm_avx512(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_uint8_float32_norm_avx512_aligned(w, h, c, src, dst);
    else
        return conv_uint8_float32_norm_avx512_unaligned(w, h, c, src, dst);
}

// USHORT
#define USHRT_DIVISOR (float)USHRT_MAX

AVX512_TGT static inline void conv_uint16_float32_norm_avx512_step(__m512i short_pack, __m512 *float_pack_lo, __m512 *float_pack_hi, __m512 scalar)
{
    __m512i int_pack_lo = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    __m512i int_pack_hi = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    *float_pack_lo = _mm512_cvtepi32_ps(int_pack_lo);
    *float_pack_hi = _mm512_cvtepi32_ps(int_pack_hi);
    *float_pack_lo = _mm512_div_ps(*float_pack_lo, scalar);
    *float_pack_hi = _mm512_div_ps(*float_pack_hi, scalar);
}

AVX512_TGT static int conv_uint16_float32_norm_avx512_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(USHRT_DIVISOR);
    __m512 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 32); idx += 32)
    {
        __m512i short_pack = _mm512_load_si512((__m512i *)&src[idx]);
        conv_uint16_float32_norm_avx512_step(short_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm512_stream_ps(&dst[idx], float_pack_lo);
        _mm512_stream_ps(&dst[idx + 16], float_pack_hi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

AVX512_TGT static int conv_uint16_float32_norm_avx512_unaligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(USHRT_DIVISOR);
    __m512 float_pack_lo, float_pack_hi;

    size_t idx = 0;
    for (; idx < (h * w * c - 32); idx += 32)
    {
        __m512i short_pack = _mm512_loadu_si512((__m512i *)&src[idx]);
        conv_uint16_float32_norm_avx512_step(short_pack, &float_pack_lo, &float_pack_hi, scalar);
        _mm512_storeu_ps(&dst[idx], float_pack_lo);
        _mm512_storeu_ps(&dst[idx + 16], float_pack_hi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint16_float32_norm_avx512(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m512));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m512));

    if (src_aligned && dst_aligned)
        return conv_uint16_float32_norm_avx512_aligned(w, h, c, src, dst);
    else
        return conv_uint16_float32_norm_avx512_unaligned(w, h, c, src, dst);
}