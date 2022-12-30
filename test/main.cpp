#include <gtest/gtest.h>
#include <new>
#include <boltimg.h>

constexpr int W = 1000;
constexpr int H = 1000;

TEST(Conversion, uint8_float32_norm)
{
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);

    uint8_t *src = new (std::align_val_t(32)) uint8_t[W * H];
    std::fill_n(src, W * H, UCHAR_MAX);
    float *dst = new (std::align_val_t(32)) float[W * H];
    EXPECT_EQ(bolt_conv_uint8_float32_norm(&ctx, W, H, 1, src, dst), BOLT_ERR_SUCCESS);

    for (size_t i = 0; i < W * H; ++i)
        EXPECT_EQ(dst[i], 1.0f);

    delete[] src;
    delete[] dst;
}

TEST(Conversion, uint16_float32_norm)
{
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);

    uint16_t *src = new (std::align_val_t(32)) uint16_t[W * H];
    std::fill_n(src, W * H, USHRT_MAX);
    float *dst = new (std::align_val_t(32)) float[W * H];
    EXPECT_EQ(bolt_conv_uint16_float32_norm(&ctx, W, H, 1, src, dst), BOLT_ERR_SUCCESS);

    for (size_t i = 0; i < W * H; ++i)
        EXPECT_EQ(dst[i], 1.0f);

    delete[] src;
    delete[] dst;
}