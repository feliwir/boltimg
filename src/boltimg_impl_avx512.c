#include "boltimg.h"
#include <immintrin.h>

// AVX2
__attribute__((__target__("avx512bw"))) static inline void conv_uint16_float32_norm_avx512_step(__m512i short_pack, __m512 *float_pack_low, __m512 *float_pack_high, __m512 scalar)
{
    __m512i int_pack_low = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    __m512i int_pack_high = _mm512_unpacklo_epi16(short_pack, _mm512_set1_epi16(0));
    *float_pack_low = _mm512_cvtepi32_ps(int_pack_low);
    *float_pack_high = _mm512_cvtepi32_ps(int_pack_high);
    *float_pack_low = _mm512_div_ps(*float_pack_low, scalar);
    *float_pack_high = _mm512_div_ps(*float_pack_high, scalar);
}

__attribute__((__target__("avx512bw"))) int conv_uint16_float32_norm_avx512_aligned(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(65535.0f);
    __m512 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w - sizeof(__m512)); idx += sizeof(__m512))
    {
        __m512i short_pack = _mm512_load_si512((__m512i *)&src[idx]);
        conv_uint16_float32_norm_avx512_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm512_stream_ps(&dst[idx], float_pack_low);
        _mm512_stream_ps(&dst[idx + 16], float_pack_high);
    }
    for (; idx < h * w; ++idx)
    {
        dst[idx] = src[idx] / 65535.0f;
    }
    return BOLT_ERR_SUCCESS;
}

__attribute__((__target__("avx512bw"))) int conv_uint16_float32_norm_avx512_unaligned(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    __m512 scalar = _mm512_set1_ps(65535.0f);
    __m512 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w - sizeof(__m512)); idx += sizeof(__m512))
    {
        __m512i short_pack = _mm512_loadu_si512((__m512i *)&src[idx]);
        conv_uint16_float32_norm_avx512_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm512_storeu_ps(&dst[idx], float_pack_low);
        _mm512_storeu_ps(&dst[idx + 16], float_pack_high);
    }
    for (; idx < h * w; ++idx)
    {
        dst[idx] = src[idx] / 65535.0f;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint16_float32_norm_avx512(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m512));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m512));

    if (src_aligned && dst_aligned)
        return conv_uint16_float32_norm_avx512_aligned(w, h, src, dst);
    else
        return conv_uint16_float32_norm_avx512_unaligned(w, h, src, dst);
}