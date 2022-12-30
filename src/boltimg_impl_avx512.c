#include "boltimg.h"
#include <immintrin.h>

#ifdef __GNUC__
#define AVX512_TGT __attribute__((__target__("avx512bw")))
#else
#define AVX512_TGT
#endif

#define USHRT_DIVISOR (float)USHRT_MAX

// AVX2
AVX512_TGT static inline void conv_uint16_float32_norm_avx512_step(__m512i short_pack, __m512 *float_pack_low, __m512 *float_pack_high, __m512 scalar)
{
    __m512i int_pack_low = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    __m512i int_pack_high = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    *float_pack_low = _mm512_cvtepi32_ps(int_pack_low);
    *float_pack_high = _mm512_cvtepi32_ps(int_pack_high);
    *float_pack_low = _mm512_div_ps(*float_pack_low, scalar);
    *float_pack_high = _mm512_div_ps(*float_pack_high, scalar);
}

AVX512_TGT static int conv_uint16_float32_norm_avx512_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(USHRT_DIVISOR);
    __m512 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 32); idx += 32)
    {
        __m512i short_pack = _mm512_load_si512((__m512i *)&src[idx]);
        conv_uint16_float32_norm_avx512_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm512_stream_ps(&dst[idx], float_pack_low);
        _mm512_stream_ps(&dst[idx + 16], float_pack_high);
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
    __m512 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w * c - 32); idx += 32)
    {
        __m512i short_pack = _mm512_loadu_si512((__m512i *)&src[idx]);
        conv_uint16_float32_norm_avx512_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm512_storeu_ps(&dst[idx], float_pack_low);
        _mm512_storeu_ps(&dst[idx + 16], float_pack_high);
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