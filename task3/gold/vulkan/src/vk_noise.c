#include "vk_noise.h"

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── compile-time knobs ───────────────────────────────────────────────────── */
#define N_BINS          100u
#define LOCAL_SIZE_X    256u    /* must match local_size_x in .comp files */
#define MAX_ERROR_LEN   512

/* ── helpers ─────────────────────────────────────────────────────────────── */
#define VK_CHECK(expr)                                              \
    do {                                                            \
        VkResult _r = (expr);                                       \
        if (_r != VK_SUCCESS) {                                     \
            snprintf(ctx->error, MAX_ERROR_LEN,                     \
                     "%s:%d  %s  = %d", __FILE__, __LINE__,         \
                     #expr, (int)_r);                               \
            return _r;                                              \
        }                                                           \
    } while (0)

#define VK_CHECK_GOTO(expr, lbl)                                    \
    do {                                                            \
        VkResult _r = (expr);                                       \
        if (_r != VK_SUCCESS) {                                     \
            snprintf(ctx->error, MAX_ERROR_LEN,                     \
                     "%s:%d  %s  = %d", __FILE__, __LINE__,         \
                     #expr, (int)_r);                               \
            goto lbl;                                               \
        }                                                           \
    } while (0)

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }


/* ══════════════════════════════════════════════════════════════════════════
 *  Pipeline handles
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    VkShaderModule        shader;
    VkDescriptorSetLayout dset_layout;
    VkPipelineLayout      pipe_layout;
    VkPipeline            pipeline;
} ComputePipe;


/* ══════════════════════════════════════════════════════════════════════════
 *  Context
 * ══════════════════════════════════════════════════════════════════════════ */

struct vk_noise_ctx {
    char error[MAX_ERROR_LEN];

    /* Core Vulkan objects */
    VkInstance                instance;
    VkPhysicalDevice          phys_dev;
    VkDevice                  device;
    uint32_t                  compute_family;
    VkQueue                   queue;

    /* Command pool / buffer (single, re-recorded each call) */
    VkCommandPool             cmd_pool;
    VkCommandBuffer           cmd_buf;
    VkFence                   fence;

    /* Descriptor pool (enough for all sets in one dispatch) */
    VkDescriptorPool          desc_pool;

    /* Three pipelines */
    ComputePipe               pipe_hist;       /* histogram.spv         */
    ComputePipe               pipe_thresh;     /* find_threshold.spv    */
    ComputePipe               pipe_apply;      /* apply_threshold.spv   */

    /* Device-local buffers (persistent, resized on demand) */
    VkBuffer                  buf_data;        /* float32 depth data    */
    VkDeviceMemory            mem_data;
    VkDeviceSize              buf_data_size;

    VkBuffer                  buf_hist;        /* uint32[N_BINS]        */
    VkDeviceMemory            mem_hist;

    VkBuffer                  buf_threshold;   /* float32 x 1           */
    VkDeviceMemory            mem_threshold;

    /* Host-visible staging buffer */
    VkBuffer                  buf_staging;
    VkDeviceMemory            mem_staging;
    VkDeviceSize              buf_staging_size;
    void                     *staging_mapped;  /* persistently mapped   */

    /* Memory type indices */
    uint32_t                  mem_device_local;
    uint32_t                  mem_host_visible;
};


/* ══════════════════════════════════════════════════════════════════════════
 *  SPIR-V loader
 * ══════════════════════════════════════════════════════════════════════════ */

static VkResult load_spirv(VkDevice device, const char *path,
                            VkShaderModule *out, char *errbuf)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(errbuf, MAX_ERROR_LEN, "Cannot open %s", path);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    uint32_t *code = malloc(sz);
    if (!code) { fclose(f); return VK_ERROR_OUT_OF_HOST_MEMORY; }
    fread(code, 1, sz, f);
    fclose(f);

    VkShaderModuleCreateInfo ci = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)sz,
        .pCode    = code,
    };
    VkResult r = vkCreateShaderModule(device, &ci, NULL, out);
    free(code);
    return r;
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Memory helper
 * ══════════════════════════════════════════════════════════════════════════ */

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                                  VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static VkResult create_buffer(vk_noise_ctx_t    *ctx,
                               VkDeviceSize       size,
                               VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags mem_props,
                               VkBuffer          *buf,
                               VkDeviceMemory    *mem)
{
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &bci, NULL, buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, *buf, &req);

    uint32_t mt = find_memory_type(ctx->phys_dev, req.memoryTypeBits, mem_props);
    if (mt == UINT32_MAX) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    VkMemoryAllocateInfo ai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = req.size,
        .memoryTypeIndex = mt,
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &ai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(ctx->device, *buf, *mem, 0));
    return VK_SUCCESS;
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Pipeline builder
 *
 *  Descriptor layout is the same for all three shaders:
 *    binding 0 → storage buffer (data / histogram / data)
 *    binding 1 → storage buffer (histogram / threshold / threshold)
 * ══════════════════════════════════════════════════════════════════════════ */

static VkResult build_pipeline(vk_noise_ctx_t *ctx,
                                const char     *spv_path,
                                uint32_t        n_bindings,      /* 1 or 2  */
                                uint32_t        push_const_size,
                                ComputePipe    *out)
{
    VK_CHECK(load_spirv(ctx->device, spv_path, &out->shader, ctx->error));

    /* Descriptor set layout */
    VkDescriptorSetLayoutBinding bindings[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = n_bindings,
        .pBindings    = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &out->dset_layout));

    /* Pipeline layout */
    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = push_const_size,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &out->dset_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pc_range,
    };
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &plci, NULL, &out->pipe_layout));

    /* Compute pipeline */
    VkComputePipelineCreateInfo cpci = {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = out->shader,
            .pName  = "main",
        },
        .layout = out->pipe_layout,
    };
    VK_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                       &out->pipeline));
    return VK_SUCCESS;
}

static void destroy_pipeline(VkDevice device, ComputePipe *p)
{
    if (p->pipeline)    vkDestroyPipeline(device, p->pipeline, NULL);
    if (p->pipe_layout) vkDestroyPipelineLayout(device, p->pipe_layout, NULL);
    if (p->dset_layout) vkDestroyDescriptorSetLayout(device, p->dset_layout, NULL);
    if (p->shader)      vkDestroyShaderModule(device, p->shader, NULL);
    memset(p, 0, sizeof(*p));
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Push-constant structs  (must match the GLSL layouts exactly)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct { uint32_t n_elements; float d_min; float d_max; uint32_t n_bins; }  PC_Hist;
typedef struct { uint32_t n_bins; float d_min; float d_max; uint32_t tail_bins; }   PC_Thresh;
typedef struct { uint32_t n_elements; }                             PC_Apply;


/* ══════════════════════════════════════════════════════════════════════════
 *  Public API: vk_noise_create
 * ══════════════════════════════════════════════════════════════════════════ */

vk_noise_ctx_t *vk_noise_create(const char *spv_dir)
{
    vk_noise_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

#define FAIL(msg) do { snprintf(ctx->error, MAX_ERROR_LEN, "%s", msg); goto fail; } while(0)
#define VKC(expr) do { if ((expr) != VK_SUCCESS) goto fail; } while(0)

    /* ── Instance ────────────────────────────────────────────────────────── */
    VkApplicationInfo app = {
        .sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo ici = {
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VKC(vkCreateInstance(&ici, NULL, &ctx->instance));

    /* ── Physical device — first with a compute queue ────────────────────── */
    uint32_t n_phys = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &n_phys, NULL);
    if (n_phys == 0) FAIL("No Vulkan-capable GPU found");

    VkPhysicalDevice *phys_devs = malloc(n_phys * sizeof(*phys_devs));
    vkEnumeratePhysicalDevices(ctx->instance, &n_phys, phys_devs);

    ctx->phys_dev       = VK_NULL_HANDLE;
    ctx->compute_family = UINT32_MAX;

    for (uint32_t i = 0; i < n_phys; i++) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_devs[i], &nq, NULL);
        VkQueueFamilyProperties *qp = malloc(nq * sizeof(*qp));
        vkGetPhysicalDeviceQueueFamilyProperties(phys_devs[i], &nq, qp);
        for (uint32_t j = 0; j < nq; j++) {
            if (qp[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx->phys_dev       = phys_devs[i];
                ctx->compute_family = j;
                break;
            }
        }
        free(qp);
        if (ctx->phys_dev != VK_NULL_HANDLE) break;
    }
    free(phys_devs);
    if (ctx->phys_dev == VK_NULL_HANDLE) FAIL("No compute-capable queue family");

    /* ── Logical device ───────────────────────────────────────────────────── */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->compute_family,
        .queueCount       = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos    = &qci,
    };
    VKC(vkCreateDevice(ctx->phys_dev, &dci, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->compute_family, 0, &ctx->queue);

    /* ── Command pool + buffer ────────────────────────────────────────────── */
    VkCommandPoolCreateInfo cpci = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->compute_family,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    VKC(vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool));

    VkCommandBufferAllocateInfo cbai = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = ctx->cmd_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VKC(vkAllocateCommandBuffers(ctx->device, &cbai, &ctx->cmd_buf));

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VKC(vkCreateFence(ctx->device, &fci, NULL, &ctx->fence));

    /* ── Descriptor pool (6 sets max per dispatch) ────────────────────────── */
    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 12,     /* 2 bindings × 6 sets */
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 6,
        .poolSizeCount = 1,
        .pPoolSizes    = &pool_size,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    };
    VKC(vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pool));

    /* ── Pipelines ────────────────────────────────────────────────────────── */
    char path[1024];

    snprintf(path, sizeof(path), "%s/histogram.spv", spv_dir);
    if (build_pipeline(ctx, path, 2, sizeof(PC_Hist), &ctx->pipe_hist) != VK_SUCCESS)
        goto fail;

    snprintf(path, sizeof(path), "%s/find_threshold.spv", spv_dir);
    if (build_pipeline(ctx, path, 2, sizeof(PC_Thresh), &ctx->pipe_thresh) != VK_SUCCESS)
        goto fail;

    snprintf(path, sizeof(path), "%s/apply_threshold.spv", spv_dir);
    if (build_pipeline(ctx, path, 2, sizeof(PC_Apply), &ctx->pipe_apply) != VK_SUCCESS)
        goto fail;

    /* ── Fixed-size device-local buffers (histogram, threshold) ──────────── */
    VKC(create_buffer(ctx,
                      N_BINS * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      &ctx->buf_hist, &ctx->mem_hist));

    VKC(create_buffer(ctx,
                      sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      &ctx->buf_threshold, &ctx->mem_threshold));

    return ctx;

fail:
    vk_noise_destroy(ctx);
    return NULL;

#undef FAIL
#undef VKC
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Ensure device + staging buffers are big enough
 * ══════════════════════════════════════════════════════════════════════════ */

static int ensure_buffers(vk_noise_ctx_t *ctx, VkDeviceSize data_bytes)
{
    /* Device-local data buffer */
    if (data_bytes > ctx->buf_data_size) {
        if (ctx->buf_data) {
            vkDestroyBuffer(ctx->device, ctx->buf_data, NULL);
            vkFreeMemory(ctx->device, ctx->mem_data, NULL);
        }
        VkResult r = create_buffer(ctx, data_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &ctx->buf_data, &ctx->mem_data);
        if (r != VK_SUCCESS) return -1;
        ctx->buf_data_size = data_bytes;
    }

    /* Staging buffer: data upload + threshold readback */
    VkDeviceSize staging_need = data_bytes > sizeof(float) ? data_bytes : sizeof(float);
    if (staging_need > ctx->buf_staging_size) {
        if (ctx->buf_staging) {
            vkUnmapMemory(ctx->device, ctx->mem_staging);
            vkDestroyBuffer(ctx->device, ctx->buf_staging, NULL);
            vkFreeMemory(ctx->device, ctx->mem_staging, NULL);
        }
        VkResult r = create_buffer(ctx, staging_need,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &ctx->buf_staging, &ctx->mem_staging);
        if (r != VK_SUCCESS) return -1;
        vkMapMemory(ctx->device, ctx->mem_staging, 0, staging_need, 0,
                    &ctx->staging_mapped);
        ctx->buf_staging_size = staging_need;
    }
    return 0;
}



/* ══════════════════════════════════════════════════════════════════════════
 *  One ABC-pass: histogram → find_threshold → apply_threshold
 *  on a sub-region [offset, offset+n_elements) of buf_data.
 * ══════════════════════════════════════════════════════════════════════════ */

static void record_one_pass(vk_noise_ctx_t *ctx,
                             VkCommandBuffer cmd,
                             uint32_t n_elements,   /* pixels in region    */
                             float    d_min,
                             float    d_max,
                             uint32_t tail_bins,
                             VkDescriptorSet dset_hist,
                             VkDescriptorSet dset_thresh,
                             VkDescriptorSet dset_apply)
{
    VkBufferMemoryBarrier bar = {
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    };

    /* ── Pass A: histogram ─────────────────────────────────────────────── */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe_hist.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipe_hist.pipe_layout, 0, 1, &dset_hist, 0, NULL);
    PC_Hist pc_hist = { n_elements, d_min, d_max, N_BINS };
    vkCmdPushConstants(cmd, ctx->pipe_hist.pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_hist), &pc_hist);
    vkCmdDispatch(cmd, ceil_div(n_elements, LOCAL_SIZE_X), 1, 1);

    /* histogram write → threshold read */
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar.buffer        = ctx->buf_hist;
    bar.size          = N_BINS * sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 1, &bar, 0, NULL);

    /* ── Pass B: find_threshold ────────────────────────────────────────── */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe_thresh.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipe_thresh.pipe_layout, 0, 1, &dset_thresh, 0, NULL);
    PC_Thresh pc_thresh = { N_BINS, d_min, d_max, tail_bins };
    vkCmdPushConstants(cmd, ctx->pipe_thresh.pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_thresh), &pc_thresh);
    vkCmdDispatch(cmd, 1, 1, 1);

    /* threshold write → apply read */
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar.buffer        = ctx->buf_threshold;
    bar.size          = sizeof(float);
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 1, &bar, 0, NULL);

    /* ── Pass C: apply_threshold ────────────────────────────────────────── */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe_apply.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipe_apply.pipe_layout, 0, 1, &dset_apply, 0, NULL);
    PC_Apply pc_apply = { n_elements};
    vkCmdPushConstants(cmd, ctx->pipe_apply.pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_apply), &pc_apply);
    vkCmdDispatch(cmd, ceil_div(n_elements, LOCAL_SIZE_X), 1, 1);
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Public API: vk_noise_remove
 * ══════════════════════════════════════════════════════════════════════════ */

int vk_noise_remove(vk_noise_ctx_t *ctx,
                    float          *data_mm,
                    uint32_t        width,
                    uint32_t        height,
                    uint32_t    strength)
{
    if (!ctx || !data_mm) return -1;

    const uint32_t total      = width * height;
    const VkDeviceSize  bytes = (VkDeviceSize)total * sizeof(float);

    /* ── 1. Ensure buffers ────────────────────────────────────────────────── */
    if (ensure_buffers(ctx, bytes) < 0) {
        snprintf(ctx->error, MAX_ERROR_LEN, "Buffer allocation failed");
        return -1;
    }

    /* ── 2. Upload data_mm → staging → device ────────────────────────────── */
    memcpy(ctx->staging_mapped, data_mm, bytes);

    /* Compute depth stats for histogram range (on CPU — tiny loop) */
    float d_min =  1e9f, d_max = -1e9f;
    for (uint32_t i = 0; i < total; i++) {
        float v = data_mm[i];
        if (v <= 0.f) continue;
        if (v < d_min) d_min = v;
        if (v > d_max) d_max = v;
    }
    if (d_min >= d_max) return 0;   /* empty or uniform crop — nothing to do */

    /* ── 3. Record command buffer ─────────────────────────────────────────── */
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);

    /* staging → device */
    VkBufferCopy copy = { 0, 0, bytes };
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->buf_staging, ctx->buf_data, 1, &copy);

    VkBufferMemoryBarrier bar = {
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer              = ctx->buf_data,
        .size                = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 1, &bar, 0, NULL);

    /* Zero histogram before first pass */
    vkCmdFillBuffer(ctx->cmd_buf, ctx->buf_hist, 0, N_BINS * sizeof(uint32_t), 0);
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    bar.buffer        = ctx->buf_hist;
    bar.size          = N_BINS * sizeof(uint32_t);
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 1, &bar, 0, NULL);

    /* Allocate 6 descriptor sets (2 per pass × 2 runs) */
    VkDescriptorSet dsets[6];
    VkDescriptorSetLayout layouts[6] = {
        ctx->pipe_hist.dset_layout,   ctx->pipe_thresh.dset_layout,
        ctx->pipe_apply.dset_layout,  ctx->pipe_hist.dset_layout,
        ctx->pipe_thresh.dset_layout, ctx->pipe_apply.dset_layout,
    };
    VkDescriptorSetAllocateInfo dsai = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = ctx->desc_pool,
        .descriptorSetCount = 6,
        .pSetLayouts        = layouts,
    };
    vkAllocateDescriptorSets(ctx->device, &dsai, dsets);

    /* Write descriptors */
    #define WRITE_BUF(dset, bind, buf, sz)                                  \
        do {                                                                 \
            VkDescriptorBufferInfo _bi = { buf, 0, sz };                    \
            VkWriteDescriptorSet   _w  = {                                  \
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  \
                .dstSet          = dset,                                     \
                .dstBinding      = bind,                                     \
                .descriptorCount = 1,                                        \
                .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,       \
                .pBufferInfo     = &_bi,                                     \
            };                                                               \
            vkUpdateDescriptorSets(ctx->device, 1, &_w, 0, NULL);          \
        } while(0)

    /* Run descriptors (full frame,) */
    WRITE_BUF(dsets[0], 0, ctx->buf_data,      bytes);
    WRITE_BUF(dsets[0], 1, ctx->buf_hist,      N_BINS * sizeof(uint32_t));
    WRITE_BUF(dsets[1], 0, ctx->buf_hist,      N_BINS * sizeof(uint32_t));
    WRITE_BUF(dsets[1], 1, ctx->buf_threshold, sizeof(float));
    WRITE_BUF(dsets[2], 0, ctx->buf_data,      bytes);
    WRITE_BUF(dsets[2], 1, ctx->buf_threshold, sizeof(float));

    /* Run full frame, bins[-strength] */
    record_one_pass(ctx, ctx->cmd_buf,
                    total,              /* n_elements */
                    d_min, d_max,
                    strength,                  /* tail_bins  */
                    dsets[0], dsets[1], dsets[2]);

    /* device → staging readback */
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar.buffer        = ctx->buf_data;
    bar.size          = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 1, &bar, 0, NULL);
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->buf_data, ctx->buf_staging, 1, &copy);

    vkEndCommandBuffer(ctx->cmd_buf);

    /* ── 4. Submit & wait ────────────────────────────────────────────────── */
    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &ctx->cmd_buf,
    };
    vkQueueSubmit(ctx->queue, 1, &si, ctx->fence);
    vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &ctx->fence);

    /* ── 5. Copy result back to caller ───────────────────────────────────── */
    memcpy(data_mm, ctx->staging_mapped, bytes);

    /* Free descriptor sets for next call */
    vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 6, dsets);

    return 0;
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Public API: vk_noise_last_error / vk_noise_destroy
 * ══════════════════════════════════════════════════════════════════════════ */

const char *vk_noise_last_error(vk_noise_ctx_t *ctx)
{
    return ctx ? ctx->error : "NULL context";
}

void vk_noise_destroy(vk_noise_ctx_t *ctx)
{
    if (!ctx) return;
    VkDevice dev = ctx->device;

    if (ctx->staging_mapped && ctx->mem_staging)
        vkUnmapMemory(dev, ctx->mem_staging);

#define DESTROY_BUF(b, m) \
    if (b) vkDestroyBuffer(dev, b, NULL); \
    if (m) vkFreeMemory(dev, m, NULL);

    DESTROY_BUF(ctx->buf_data,      ctx->mem_data)
    DESTROY_BUF(ctx->buf_hist,      ctx->mem_hist)
    DESTROY_BUF(ctx->buf_threshold, ctx->mem_threshold)
    DESTROY_BUF(ctx->buf_staging,   ctx->mem_staging)
#undef DESTROY_BUF

    destroy_pipeline(dev, &ctx->pipe_hist);
    destroy_pipeline(dev, &ctx->pipe_thresh);
    destroy_pipeline(dev, &ctx->pipe_apply);

    if (ctx->desc_pool) vkDestroyDescriptorPool(dev, ctx->desc_pool, NULL);
    if (ctx->fence)     vkDestroyFence(dev, ctx->fence, NULL);
    if (ctx->cmd_pool)  vkDestroyCommandPool(dev, ctx->cmd_pool, NULL);
    if (dev)            vkDestroyDevice(dev, NULL);
    if (ctx->instance)  vkDestroyInstance(ctx->instance, NULL);

    free(ctx);
}
