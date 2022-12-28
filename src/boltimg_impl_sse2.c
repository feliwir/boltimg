#include "boltimg.h"
#include <emmintrin.h>

// SSE2
__attribute__((__target__("sse2"))) static inline void conv_uint16_float32_norm_sse2_step(__m128i short_pack, __m128 *float_pack_low, __m128 *float_pack_high, __m128 scalar)
{
    __m128i int_pack_low = _mm_unpacklo_epi16(short_pack, _mm_set1_epi16(0));
    __m128i int_pack_high = _mm_unpackhi_epi16(short_pack, _mm_set1_epi16(0));
    *float_pack_low = _mm_cvtepi32_ps(int_pack_low);
    *float_pack_high = _mm_cvtepi32_ps(int_pack_high);
    *float_pack_low = _mm_div_ps(*float_pack_low, scalar);
    *float_pack_high = _mm_div_ps(*float_pack_high, scalar);
}

__attribute__((__target__("sse2"))) int conv_uint16_float32_norm_sse2_aligned(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(65535.0f);
    __m128 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w - sizeof(__m128)); idx += sizeof(__m128))
    {
        __m128i short_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_uint16_float32_norm_sse2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm_stream_ps(&dst[idx], float_pack_low);
        _mm_stream_ps(&dst[idx + 4], float_pack_high);
    }
    for (; idx < h * w; ++idx)
    {
        dst[idx] = src[idx] / 65535.0f;
    }
    return BOLT_ERR_SUCCESS;
}

__attribute__((__target__("sse2"))) int conv_uint16_float32_norm_sse2_unaligned(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(65535.0f);
    __m128 float_pack_low, float_pack_high;

    size_t idx = 0;
    for (; idx < (h * w - sizeof(__m128)); idx += sizeof(__m128))
    {
        __m128i short_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_uint16_float32_norm_sse2_step(short_pack, &float_pack_low, &float_pack_high, scalar);
        _mm_storeu_ps(&dst[idx], float_pack_low);
        _mm_storeu_ps(&dst[idx + 4], float_pack_high);
    }
    for (; idx < h * w; ++idx)
    {
        dst[idx] = src[idx] / 65535.0f;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_uint16_float32_norm_sse2(size_t w, size_t h, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_uint16_float32_norm_sse2_aligned(w, h, src, dst);
    else
        return conv_uint16_float32_norm_sse2_unaligned(w, h, src, dst);
}