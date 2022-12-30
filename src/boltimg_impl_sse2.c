#include "boltimg.h"
#include <emmintrin.h>

#ifdef __GNUC__
#define SSE2_TGT __attribute__((__target__("sse2")))
#else
#define SSE2_TGT
#endif

// UINT8
#define UCHAR_DIVISOR (float)UCHAR_MAX

SSE2_TGT static inline void conv_uint8_float32_norm_sse2_step(__m128i byte_pack, __m128 *float_pack_low, __m128 *float_pack_high, __m128 scalar)
{
    // TODO:
    __m128i int_pack_low = _mm_unpacklo_epi8(byte_pack, _mm_set1_epi8(0));
    __m128i int_pack_high = _mm_unpackhi_epi8(byte_pack, _mm_set1_epi8(0));
    *float_pack_low = _mm_cvtepi32_ps(int_pack_low);
    *float_pack_high = _mm_cvtepi32_ps(int_pack_high);
    *float_pack_low = _mm_div_ps(*float_pack_low, scalar);
    *float_pack_high = _mm_div_ps(*float_pack_high, scalar);
}

SSE2_TGT static int conv_uint8_float32_norm_sse2_aligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(UCHAR_DIVISOR);
    __m128 float_pack_low,float_pack_mlow, float_pack_mhigh, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_uint8_float32_norm_sse2_step(byte_pack, &float_pack_low, &float_pack_high, scalar);
        _mm_stream_ps(&dst[idx], float_pack_low);
        _mm_stream_ps(&dst[idx + 8], float_pack_high);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

SSE2_TGT static int conv_uint8_float32_norm_sse2_unaligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(UCHAR_DIVISOR);
    __m128 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i short_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_uint8_float32_norm_sse2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm_storeu_ps(&dst[idx], float_pack_low);
        _mm_storeu_ps(&dst[idx + 4], float_pack_high);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint8_float32_norm_sse2(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_uint8_float32_norm_sse2_aligned(w, h, c, src, dst);
    else
        return conv_uint8_float32_norm_sse2_unaligned(w, h, c, src, dst);
}

// UINT16
#define USHRT_DIVISOR (float)USHRT_MAX

SSE2_TGT static inline void conv_uint16_float32_norm_sse2_step(__m128i short_pack, __m128 *float_pack_low, __m128 *float_pack_high, __m128 scalar)
{
    __m128i int_pack_low = _mm_unpacklo_epi16(short_pack, _mm_set1_epi16(0));
    __m128i int_pack_high = _mm_unpackhi_epi16(short_pack, _mm_set1_epi16(0));
    *float_pack_low = _mm_cvtepi32_ps(int_pack_low);
    *float_pack_high = _mm_cvtepi32_ps(int_pack_high);
    *float_pack_low = _mm_div_ps(*float_pack_low, scalar);
    *float_pack_high = _mm_div_ps(*float_pack_high, scalar);
}

SSE2_TGT static int conv_uint16_float32_norm_sse2_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(USHRT_DIVISOR);
    __m128 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 8); idx += 8)
    {
        __m128i short_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_uint16_float32_norm_sse2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm_stream_ps(&dst[idx], float_pack_low);
        _mm_stream_ps(&dst[idx + 4], float_pack_high);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

SSE2_TGT static int conv_uint16_float32_norm_sse2_unaligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(USHRT_DIVISOR);
    __m128 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 8); idx += 8)
    {
        __m128i short_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_uint16_float32_norm_sse2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm_storeu_ps(&dst[idx], float_pack_low);
        _mm_storeu_ps(&dst[idx + 4], float_pack_high);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint16_float32_norm_sse2(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_uint16_float32_norm_sse2_aligned(w, h, c, src, dst);
    else
        return conv_uint16_float32_norm_sse2_unaligned(w, h, c, src, dst);
}