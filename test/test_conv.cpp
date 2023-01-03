#include <gtest/gtest.h>
#include <new>
#include <boltimg.h>

constexpr int W = 1000;
constexpr int H = 1000;

TEST(Conversion, uint8_f32_norm)
{
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);

    uint8_t *src = static_cast<uint8_t *>(bolt_alloc(sizeof(uint8_t) * W * H));
    std::fill_n(src, W * H, UCHAR_MAX);
    float *dst = static_cast<float *>(bolt_alloc(sizeof(float) * W * H));
    EXPECT_EQ(bolt_conv_u8_f32_norm(&ctx, W, H, 1, src, dst), BOLT_ERR_SUCCESS);

    for (size_t i = 0; i < W * H; ++i)
        EXPECT_EQ(dst[i], 1.0f);

    bolt_free(src);
    bolt_free(dst);
}

TEST(Conversion, uint16_f32_norm)
{
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);

    uint16_t *src = static_cast<uint16_t *>(bolt_alloc(sizeof(uint16_t) * W * H));
    std::fill_n(src, W * H, USHRT_MAX);
    float *dst = static_cast<float *>(bolt_alloc(sizeof(float) * W * H));
    EXPECT_EQ(bolt_conv_u16_f32_norm(&ctx, W, H, 1, src, dst), BOLT_ERR_SUCCESS);

    for (size_t i = 0; i < W * H; ++i)
        EXPECT_EQ(dst[i], 1.0f);

    bolt_free(src);
    bolt_free(dst);
}

#define STBI_MALLOC bolt_alloc
#define STBI_FREE bolt_free
#define STBI_REALLOC bolt_realloc

#define STB_IMAGE_IMPLEMENTATION
#include "../deps/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../deps/stb_image_write.h"

#define CONCAT(A, B) A "" B

#ifndef BOLTIMG_TEST_ASSETS
#error "Missing macro"
#endif

TEST(Conversion, load_img_rgb8)
{
    int w = 0, h = 0, c = 0;
    stbi_uc *rgb8 = stbi_load(CONCAT(BOLTIMG_TEST_ASSETS, "lena.png"), &w, &h, &c, 3);
    ASSERT_NE(rgb8, nullptr);
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);
    float *rgbf32 = static_cast<float *>(bolt_alloc(sizeof(float) * w * h * c));
    EXPECT_EQ(bolt_conv_u8_f32_norm(&ctx, w, h, c, rgb8, rgbf32), BOLT_ERR_SUCCESS);

    EXPECT_NE(stbi_write_hdr("lena8.hdr", w, h, c, rgbf32), 0);

    bolt_free(rgbf32);
    stbi_image_free(rgb8);
}

TEST(Conversion, load_img_rgb16)
{
    int w = 0, h = 0, c = 0;
    stbi_us *rgb16 = stbi_load_16(CONCAT(BOLTIMG_TEST_ASSETS, "lena.png"), &w, &h, &c, 3);
    ASSERT_NE(rgb16, nullptr);
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);
    float *rgbf32 = static_cast<float *>(bolt_alloc(sizeof(float) * w * h * c));
    EXPECT_EQ(bolt_conv_u16_f32_norm(&ctx, w, h, c, rgb16, rgbf32), BOLT_ERR_SUCCESS);

    EXPECT_NE(stbi_write_hdr("lena16.hdr", w, h, c, rgbf32), 0);

    bolt_free(rgbf32);
    stbi_image_free(rgb16);
}