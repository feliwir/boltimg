#include "boltimg.h"
#include <smmintrin.h>

#ifdef __GNUC__
#define SSE4_TGT __attribute__((__target__("sse4")))
#else
#define SSE4_TGT
#endif

// UINT8
#define UCHAR_DIVISOR (float)UCHAR_MAX

SSE4_TGT static inline void conv_uint8_float32_norm_sse4_step(__m128i byte_pack, __m128 *float_pack_lolo, __m128 *float_pack_lo, __m128 *float_pack_hi, __m128 *float_pack_hihi, __m128 scalar)
{
    __m128i short_pack_lo = _mm_cvtepu8_epi16(byte_pack);
    __m128i shuffle8 = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i hihi8 = _mm_shuffle_epi8(byte_pack, shuffle8);
    __m128i short_pack_hi = _mm_cvtepu8_epi16(hihi8);
    __m128i int_pack_lolo = _mm_unpacklo_epi16(short_pack_lo, _mm_set1_epi16(0));
    __m128i int_pack_lo = _mm_unpackhi_epi16(short_pack_lo, _mm_set1_epi16(0));
    __m128i int_pack_hi = _mm_unpacklo_epi16(short_pack_hi, _mm_set1_epi16(0));
    __m128i int_pack_hihi = _mm_unpackhi_epi16(short_pack_hi, _mm_set1_epi16(0));

    *float_pack_lolo = _mm_cvtepi32_ps(int_pack_lolo);
    *float_pack_lo = _mm_cvtepi32_ps(int_pack_lo);
    *float_pack_hi = _mm_cvtepi32_ps(int_pack_hi);
    *float_pack_hihi = _mm_cvtepi32_ps(int_pack_hihi);

    *float_pack_lolo = _mm_div_ps(*float_pack_lolo, scalar);
    *float_pack_lo = _mm_div_ps(*float_pack_lo, scalar);
    *float_pack_hi = _mm_div_ps(*float_pack_hi, scalar);
    *float_pack_hihi = _mm_div_ps(*float_pack_hihi, scalar);
}

SSE4_TGT static int conv_uint8_float32_norm_sse4_aligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(UCHAR_DIVISOR);
    __m128 float_pack_lolo, float_pack_lo, float_pack_hi, float_pack_hihi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_uint8_float32_norm_sse4_step(byte_pack, &float_pack_lolo, &float_pack_lo, &float_pack_hi, &float_pack_hihi, scalar);
        _mm_stream_ps(&dst[idx], float_pack_lolo);
        _mm_stream_ps(&dst[idx + 4], float_pack_lo);
        _mm_stream_ps(&dst[idx + 8], float_pack_hi);
        _mm_stream_ps(&dst[idx + 12], float_pack_hihi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

SSE4_TGT static int conv_uint8_float32_norm_sse4_unaligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(UCHAR_DIVISOR);
    __m128 float_pack_lolo, float_pack_lo, float_pack_hi, float_pack_hihi;

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_uint8_float32_norm_sse4_step(byte_pack, &float_pack_lolo, &float_pack_lo, &float_pack_hi, &float_pack_hihi, scalar);
        _mm_storeu_ps(&dst[idx], float_pack_lolo);
        _mm_storeu_ps(&dst[idx + 4], float_pack_lo);
        _mm_storeu_ps(&dst[idx + 8], float_pack_hi);
        _mm_storeu_ps(&dst[idx + 12], float_pack_hihi);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint8_float32_norm_sse4(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_uint8_float32_norm_sse4_aligned(w, h, c, src, dst);
    else
        return conv_uint8_float32_norm_sse4_unaligned(w, h, c, src, dst);
}
