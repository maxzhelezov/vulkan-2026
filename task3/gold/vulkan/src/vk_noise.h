#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vk_noise_ctx vk_noise_ctx_t;

// Initialises Vulkan instance + device + the three compute pipelines.
vk_noise_ctx_t *vk_noise_create(const char *spv_dir);

int vk_noise_remove(vk_noise_ctx_t *ctx,
                    float          *data_mm,
                    uint32_t        width,
                    uint32_t        height,
                    uint32_t strength);

/* Returns a static string describing the last error, or NULL. */
const char *vk_noise_last_error(vk_noise_ctx_t *ctx);

/* Free all Vulkan resources. */
void vk_noise_destroy(vk_noise_ctx_t *ctx);
