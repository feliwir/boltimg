#include <gtest/gtest.h>
#include <new>
#include <boltimg.h>

constexpr int W = 10000;
constexpr int H = 10000;

TEST(Conversion, uint16_float32_norm)
{
    BoltContext ctx;
    EXPECT_EQ(bolt_ctx_init(&ctx, BOLT_HL_AUTO), BOLT_ERR_SUCCESS);

    uint16_t *src_a = new (std::align_val_t(32)) uint16_t[W * H];
    float *dst_a = new (std::align_val_t(32)) float[W * H];
    EXPECT_EQ(bolt_conv_uint16_float32_norm(&ctx, W, H, 1, src_a, dst_a), BOLT_ERR_SUCCESS);

    delete[] src_a;
    delete[] dst_a;

    uint16_t *src_u = new (std::align_val_t(1)) uint16_t[W * H];
    float *dst_u = new (std::align_val_t(1)) float[W * H];
    EXPECT_EQ(bolt_conv_uint16_float32_norm(&ctx, W, H, 1, src_u, dst_u), BOLT_ERR_SUCCESS);

    delete[] src_u;
    delete[] dst_u;
}