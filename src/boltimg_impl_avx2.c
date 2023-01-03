#include "boltimg.h"
#include <immintrin.h>

#ifdef __GNUC__
#define AVX2_TGT __attribute__((__target__("avx2")))
#else
#define AVX2_TGT
#endif

// UINT8
#define UCHAR_DIVISOR (float)UCHAR_MAX

AVX2_TGT static inline void convert_u8x16_to_u32x16(__m128i bytes, __m256i result[2])
{
    result[0] = _mm256_cvtepu8_epi32(bytes);
    __m128i mask = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i u8x8 = _mm_shuffle_epi8(bytes, mask);
    result[1] = _mm256_cvtepu8_epi32(u8x8);
}

AVX2_TGT static inline void conv_u8_f32_norm_avx2_step(__m128i byte_pack, __m256 float_pack[2], __m256 scalar)
{
    __m256i int_pack[2];
    convert_u8x16_to_u32x16(byte_pack, int_pack);

    for (int i = 0; i < 2; i++)
    {
        __m256 unscaled = _mm256_cvtepi32_ps(int_pack[i]);
        float_pack[i] = _mm256_div_ps(unscaled, scalar);
    }
}

AVX2_TGT static int conv_u8_f32_norm_avx2_aligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(UCHAR_DIVISOR);
    __m256 float_pack[2];

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_u8_f32_norm_avx2_step(byte_pack, float_pack, scalar);
        _mm256_stream_ps(&dst[idx], float_pack[0]);
        _mm256_stream_ps(&dst[idx + 8], float_pack[1]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

AVX2_TGT static int conv_u8_f32_norm_avx2_unaligned(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(UCHAR_DIVISOR);
    __m256 float_pack[2];

    size_t idx = 0;
    for (; idx < (h * w * c - 16); idx += 16)
    {
        __m128i byte_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_u8_f32_norm_avx2_step(byte_pack, float_pack, scalar);
        _mm256_storeu_ps(&dst[idx], float_pack[0]);
        _mm256_storeu_ps(&dst[idx + 8], float_pack[1]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / UCHAR_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_u8_f32_norm_avx2(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m256));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m256));

    if (src_aligned && dst_aligned)
        return conv_u8_f32_norm_avx2_aligned(w, h, c, src, dst);
    else
        return conv_u8_f32_norm_avx2_unaligned(w, h, c, src, dst);
}

// UINT16
#define USHRT_DIVISOR (float)USHRT_MAX

AVX2_TGT static inline void conv_u16_f32_norm_avx2_step(__m128i short_pack, __m256 float_pack[1], __m256 scalar)
{
    __m256i int_pack = _mm256_cvtepu16_epi32(short_pack);
    __m256 unscaled = _mm256_cvtepi32_ps(int_pack);
    float_pack[0] = _mm256_div_ps(unscaled, scalar);
}

AVX2_TGT static int conv_u16_f32_norm_avx2_aligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(USHRT_DIVISOR);
    __m256 float_pack[1];

    size_t idx = 0;
    for (; idx < (h * w * c - 8); idx += 8)
    {
        __m128i short_pack = _mm_load_si128((__m128i *)&src[idx]);
        conv_u16_f32_norm_avx2_step(short_pack, float_pack, scalar);
        _mm256_stream_ps(&dst[idx], float_pack[0]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

AVX2_TGT static int conv_u16_f32_norm_avx2_unaligned(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    __m256 scalar = _mm256_set1_ps(USHRT_DIVISOR);
    __m256 float_pack[1];

    size_t idx = 0;
    for (; idx < (h * w * c - 8); idx += 8)
    {
        __m128i short_pack = _mm_loadu_si128((__m128i *)&src[idx]);
        conv_u16_f32_norm_avx2_step(short_pack, float_pack, scalar);
        _mm256_storeu_ps(&dst[idx], float_pack[0]);
    }
    for (; idx < h * w * c; ++idx)
    {
        dst[idx] = src[idx] / USHRT_DIVISOR;
    }
    return BOLT_ERR_SUCCESS;
}

int conv_u16_f32_norm_avx2(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst)
{
    bool src_aligned = bolt_is_aligned(src, sizeof(__m128));
    bool dst_aligned = bolt_is_aligned(dst, sizeof(__m256));

    if (src_aligned && dst_aligned)
        return conv_u16_f32_norm_avx2_aligned(w, h, c, src, dst);
    else
        return conv_u16_f32_norm_avx2_unaligned(w, h, c, src, dst);
}