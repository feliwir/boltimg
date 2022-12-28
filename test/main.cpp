#include <gtest/gtest.h>
#include <new>
#include <boltimg.h>

TEST(Conversion, uint16_float32_norm)
{
    BoltContext ctx;
    bolt_ctx_init(&ctx, BOLT_HL_AUTO);

    uint16_t *src = new (std::align_val_t(32)) uint16_t[100 * 100];
    float *dst = new (std::align_val_t(32)) float[100 * 100];
    bolt_conv_uint16_float32_norm(&ctx, 100, 100, src, dst);

    delete[] src;
    delete[] dst;
}