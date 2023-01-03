#include "boltimg.h"
#include <smmintrin.h>

#ifdef __GNUC__
#define SSE4_TGT __attribute__((__target__("sse4")))
#else
#define SSE4_TGT
#endif

// UINT8
#define UCHAR_DIVISOR (float)UCHAR_MAX

SSE4_TGT static inline void convert_u8x16_to_u32x16(__m128i bytes, __m128i result[4])
{
    result[0] = _mm_cvtepu8_epi32(bytes);

    for (int i = 1; i < 4; i++)
    {
        __m128i mask = _mm_setr_epi8(i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3);
        __m128i i8x4 = _mm_shuffle_epi8(bytes, mask);
        result[i] = _mm_cvtepu8_epi32(i8x4);
    }
}

SSE4_TGT static inline void conv_u8_f32_norm_sse4_step(__m128i byte_pack, __m128 float_pack[4], __m128 scalar)
{
    __m128i int_pack[4];
    convert_u8x16_to_u32x16(byte_pack, int_pack);

    for (int i = 0; i < 4; i++)
    {
        __m128 unscaled = _mm_cvtepi32_ps(int_pack[i]);
        float_pack[i] = _mm_div_ps(unscaled, scalar);
    }
}

SSE4_TGT static int conv_u8_f32_norm_sse4_aligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(UCHAR_DIVISOR);
    __m128 float_pack[4];

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_u8_f32_norm_sse4_step(byte_pack, float_pack, scalar);
        for (size_t i = 0; i < 4; ++i)
            _mm_stream_ps(&dst[idx + i * 4], float_pack[i]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

SSE4_TGT static int conv_u8_f32_norm_sse4_unaligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(UCHAR_DIVISOR);
    __m128 float_pack[4];

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_u8_f32_norm_sse4_step(byte_pack, float_pack, scalar);
       for (size_t i = 0; i < 4; ++i)
            _mm_storeu_ps(&dst[idx + i * 4], float_pack[i]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_u8_f32_norm_sse4(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_u8_f32_norm_sse4_aligned(w, h, c, src, dst);
    else
        return conv_u8_f32_norm_sse4_unaligned(w, h, c, src, dst);
}

// UINT16
#define USHRT_DIVISOR (float)USHRT_MAX

SSE4_TGT static inline void conv_u16_f32_norm_sse4_step(__m128i short_pack, __m128 float_pack[2], __m128 scalar)
{
    __m128i int_pack[2] = {_mm_cvtepu16_epi32(short_pack), _mm_cvtepu16_epi32(_mm_unpackhi_epi64(short_pack, short_pack))};
    for(int i = 0;i<2;++i)
    {
        __m128 unscaled = _mm_cvtepi32_ps(int_pack[i]);
        float_pack[i] = _mm_div_ps(unscaled, scalar);
    }
}

SSE4_TGT static int conv_u16_f32_norm_sse4_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(USHRT_DIVISOR);
    __m128 float_pack[2];

    size_t idx = 0;
    for (; idx < (h * w * c - 8); idx += 8)
    {
        __m128i short_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_u16_f32_norm_sse4_step(short_pack, float_pack, scalar);
        _mm_stream_ps(&dst[idx], float_pack[0]);
        _mm_stream_ps(&dst[idx + 4], float_pack[1]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

SSE4_TGT static int conv_u16_f32_norm_sse4_unaligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m128 scalar = _mm_set1_ps(USHRT_DIVISOR);
    __m128 float_pack[2];

    size_t idx = 0;
    for (; idx < (h * w * c - 8); idx += 8)
    {
        __m128i short_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_u16_f32_norm_sse4_step(short_pack, float_pack, scalar);
        _mm_storeu_ps(&dst[idx], float_pack[0]);
        _mm_storeu_ps(&dst[idx + 4], float_pack[1]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_u16_f32_norm_sse4(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m128));

    if (src_aligned && dst_aligned)
        return conv_u16_f32_norm_sse4_aligned(w, h, c, src, dst);
    else
        return conv_u16_f32_norm_sse4_unaligned(w, h, c, src, dst);
}