#include "boltimg.h"

#if defined(__x86_64__) || defined(_M_AMD64)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
#include <cpuid.h>
#endif

static void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx,
                  uint32_t *edx)
{

#if defined(_MSC_VER)
    int cpu_info[4];
    __cpuid(cpu_info, *eax);
    *eax = cpu_info[0];
    *ebx = cpu_info[1];
    *ecx = cpu_info[2];
    *edx = cpu_info[3];
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
    uint32_t level = *eax;
    __get_cpuid(level, eax, ebx, ecx, edx);
#else
    uint32_t a = *eax, b, c = *ecx, d;
    __asm__("cpuid\n\t"
            : "+a"(a), "=b"(b), "+c"(c), "=d"(d));
    *eax = a;
    *ebx = b;
    *ecx = c;
    *edx = d;
#endif
}
#endif

extern int conv_uint8_float32_norm_avx512(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst);
extern int conv_uint8_float32_norm_avx2(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst);
extern int conv_uint8_float32_norm_sse4(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst);
extern int conv_uint8_float32_norm_scalar(size_t w, size_t h, size_t c, uint8_t *restrict src, float *restrict dst);

extern int conv_uint16_float32_norm_avx512(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst);
extern int conv_uint16_float32_norm_avx2(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst);
extern int conv_uint16_float32_norm_sse2(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst);
extern int conv_uint16_float32_norm_scalar(size_t w, size_t h, size_t c, uint16_t *restrict src, float *restrict dst);

static bool bolt_ctx_setup_dispatch(BoltContext *ctx, BoltHardwareLevel hl)
{
    switch (hl)
    {
    case BOLT_HL_AVX512:
        //ctx->conv_uint8_float32_norm = conv_uint8_float32_norm_avx512;
        ctx->conv_uint16_float32_norm = conv_uint16_float32_norm_avx512;
        break;
    case BOLT_HL_AVX2:
        ctx->conv_uint8_float32_norm = conv_uint8_float32_norm_avx2;
        ctx->conv_uint16_float32_norm = conv_uint16_float32_norm_avx2;
        break;
    case BOLT_HL_AVX:
    case BOLT_HL_SSE4:
        ctx->conv_uint8_float32_norm = conv_uint8_float32_norm_sse4;
    case BOLT_HL_SSE2:
        ctx->conv_uint8_float32_norm = conv_uint8_float32_norm_sse4;
        ctx->conv_uint16_float32_norm = conv_uint16_float32_norm_sse2;
        break;
    case BOLT_HL_SCALAR: // Stick with the scalar version we already had
        break;
    default:
        return false;
    }
    return true;
}

int bolt_ctx_init(BoltContext *ctx, BoltHardwareLevel target_lvl)
{
    // Setup scalar versions first
    ctx->conv_uint8_float32_norm = conv_uint8_float32_norm_scalar;
    ctx->conv_uint16_float32_norm = conv_uint16_float32_norm_scalar;

    // Try to find the best vectorized version now
    BoltHardwareLevel selected_lvl;
#if defined(__x86_64__) || defined(_M_AMD64)
    if (target_lvl == BOLT_HL_NEON)
        return BOLT_ERR_UNSUPPORTED;

    // Check SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 and AVX support
    uint32_t eax = 1, ebx, ecx, edx;
    cpuid(&eax, &ebx, &ecx, &edx);
    bool HAS_SSE = edx & (1 << 25) || false;
    bool HAS_SSE2 = edx & (1 << 26) || false;
    bool HAS_SSE3 = ecx & (1 << 0) || false;
    bool HAS_SSSE3 = ecx & (1 << 9) || false;
    bool HAS_SSE4_1 = ecx & (1 << 19) || false;
    bool HAS_SSE4_2 = ecx & (1 << 20) || false;
    bool HAS_AVX = ecx & (1 << 28) || false;
    // Check AVX2 and AVX512 support
    eax = 0x07, ecx = 0x0;
    cpuid(&eax, &ebx, &ecx, &edx);
    bool HAS_AVX2 = ebx & (1 << 5) || false;
    bool HAS_AVX512 = ebx & (1 << 30) || false; // WE NEED AVX512-BW

    if (target_lvl == BOLT_HL_AUTO)
    {
        selected_lvl = HAS_AVX512 ? BOLT_HL_AVX512 : HAS_AVX2 ? BOLT_HL_AVX2
                                                 : HAS_AVX    ? BOLT_HL_AVX
                                                 : HAS_SSE4_1 ? BOLT_HL_SSE4
                                                 : HAS_SSE2   ? BOLT_HL_SSE2
                                                              : BOLT_HL_SCALAR;
    }
    else if (target_lvl == BOLT_HL_AVX512 && !HAS_AVX512)
        return BOLT_ERR_UNSUPPORTED;
    else if (target_lvl == BOLT_HL_AVX && !HAS_AVX)
        return BOLT_ERR_UNSUPPORTED;
    else if (target_lvl == BOLT_HL_SSE4 && !HAS_SSE4_1)
        return BOLT_ERR_UNSUPPORTED;
    else if (target_lvl == BOLT_HL_SSE2 && !HAS_SSE2)
        return BOLT_ERR_UNSUPPORTED;
    else
        selected_lvl = target_lvl;
#endif

    if (!bolt_ctx_setup_dispatch(ctx, selected_lvl))
        return BOLT_ERR_DISPATCH_FAILED;

    return BOLT_ERR_SUCCESS;
}

bool bolt_is_aligned(const void *ptr, size_t alignment)
{
    return (((uintptr_t)ptr) % (alignment) == 0);
}

int bolt_conv_uint8_float32_norm(BoltContext *ctx, size_t w, size_t h, size_t c, uint8_t *src, float *dst)
{
    return ctx->conv_uint8_float32_norm(w, h, c, src, dst);
}

int bolt_conv_uint16_float32_norm(BoltContext *ctx, size_t w, size_t h, size_t c, uint16_t *src, float *dst)
{
    return ctx->conv_uint16_float32_norm(w, h, c, src, dst);
}