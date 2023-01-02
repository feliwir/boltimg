#pragma once
#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

typedef enum _BoltError
{
    BOLT_ERR_SUCCESS = 0,
    BOLT_ERR_UNSUPPORTED = -1,
    BOLT_ERR_DISPATCH_FAILED = -2,
} BoltError;

typedef enum _BoltHardwareLevel
{
    BOLT_HL_AUTO,
    BOLT_HL_SCALAR,
    BOLT_HL_SSE2,
    BOLT_HL_SSE4,
    BOLT_HL_AVX,
    BOLT_HL_AVX2,
    BOLT_HL_AVX512,
    BOLT_HL_NEON,
} BoltHardwareLevel;

typedef struct _BoltContext
{
    int (*conv_uint8_float32_norm)(size_t w, size_t h, size_t c, uint8_t *src, float *dst);
    int (*conv_uint16_float32_norm)(size_t w, size_t h, size_t c, uint16_t *src, float *dst);
} BoltContext;

int bolt_ctx_init(BoltContext *ctx, BoltHardwareLevel hl);
int bolt_conv_uint8_float32_norm(BoltContext *ctx, size_t w, size_t h, size_t c, uint8_t *src, float *dst);
int bolt_conv_uint16_float32_norm(BoltContext *ctx, size_t w, size_t h, size_t c, uint16_t *src, float *dst);

bool bolt_is_aligned(const void *ptr, size_t alignment);

#ifdef __cplusplus
}
#endif