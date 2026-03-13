#include "ggml-fpga.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <linux/types.h>

// --- FPGA Hardware Definitions ---

#define QK_K 256

typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[96];
    uint8_t  qs[1024];
} block_q4_Kx8_kernel;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K_kernel;

typedef struct {
    int n;
    size_t bs;
    int nr;
    int nc;
} gemv_config;

#define FPGA_IOCTL_GEMV _IOWR('L', 2, gemv_config *)

// --- Backend Implementation ---

static int g_fpga_fd = -1;

// Memory offsets as per hardware specification
static const off_t FPGA_OFFSET_LUT = 0;
static const off_t FPGA_OFFSET_S   = 64LL * 1024 * 1024;
static const off_t FPGA_OFFSET_VX  = 128LL * 1024 * 1024;
static const off_t FPGA_OFFSET_VY  = 144LL * 1024 * 1024;

// FPGA Backend Context
struct ggml_backend_fpga_context {
    std::string device_path;
    int fd = -1;

    ggml_backend_fpga_context(const char * path) : device_path(path) {
        fd = open(path, O_RDWR);
        if (fd < 0) {
            GGML_LOG_ERROR("%s: failed to open FPGA device %s\n", __func__, path);
        } else {
            GGML_LOG_INFO("%s: FPGA device initialized at %s (fd=%d)\n", __func__, path, fd);
            g_fpga_fd = fd;
        }
    }

    ~ggml_backend_fpga_context() {
        if (fd >= 0) {
            close(fd);
            if (g_fpga_fd == fd) g_fpga_fd = -1;
        }
    }
};

struct ggml_backend_fpga_buffer_context {
    void * data;
    size_t size;
};

static void ggml_backend_fpga_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_fpga_buffer_context * ctx = (ggml_backend_fpga_buffer_context *)buffer->context;
    free(ctx->data);
    delete ctx;
}

static void * ggml_backend_fpga_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_fpga_buffer_context * ctx = (ggml_backend_fpga_buffer_context *)buffer->context;
    return ctx->data;
}

static void ggml_backend_fpga_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(tensor);
    ggml_backend_fpga_buffer_context * ctx = (ggml_backend_fpga_buffer_context *)buffer->context;
    memcpy((char *)ctx->data + offset, data, size);
}

static void ggml_backend_fpga_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(tensor);
    ggml_backend_fpga_buffer_context * ctx = (ggml_backend_fpga_buffer_context *)buffer->context;
    memcpy(data, (const char *)ctx->data + offset, size);
}

static bool ggml_backend_fpga_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_fpga_buffer_set_tensor(buffer, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_fpga_buffer_get_tensor(buffer, src, dst->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_fpga_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_fpga_buffer_context * ctx = (ggml_backend_fpga_buffer_context *)buffer->context;
    memset(ctx->data, value, ctx->size);
}

static ggml_backend_buffer_i ggml_backend_fpga_buffer_interface = {
    /* .free_buffer      = */ ggml_backend_fpga_buffer_free_buffer,
    /* .get_base         = */ ggml_backend_fpga_buffer_get_base,
    /* .init_tensor      = */ NULL,
    /* .memset_tensor    = */ NULL,
    /* .set_tensor       = */ ggml_backend_fpga_buffer_set_tensor,
    /* .get_tensor       = */ ggml_backend_fpga_buffer_get_tensor,
    /* .cpy_tensor       = */ ggml_backend_fpga_buffer_cpy_tensor,
    /* .clear            = */ ggml_backend_fpga_buffer_clear,
    /* .reset            = */ NULL,
};

static const char * ggml_backend_fpga_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "FPGA";
}

static ggml_backend_buffer_t ggml_backend_fpga_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = malloc(size);
    if (!data) return nullptr;
    ggml_backend_fpga_buffer_context * ctx = new ggml_backend_fpga_buffer_context{data, size};
    return ggml_backend_buffer_init(buft, ggml_backend_fpga_buffer_interface, ctx, size);
}

static size_t ggml_backend_fpga_buffer_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static const char * ggml_backend_fpga_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FPGA";
}

static void ggml_backend_fpga_free(ggml_backend_t backend) {
    ggml_backend_fpga_context * ctx = (ggml_backend_fpga_context *)backend->context;
    delete ctx;
    delete backend;
}

bool ggml_fpga_gemv_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    static int offload_enabled = -1;
    if (offload_enabled < 0) {
        const char * env = std::getenv("GGML_FPGA_OFFLOAD");
        offload_enabled = (env != nullptr && std::string(env) == "1");
    }

    if (!offload_enabled) {
        return false;
    }

    if (g_fpga_fd < 0) {
        g_fpga_fd = open("/dev/fpga0", O_RDWR);
        if (g_fpga_fd < 0) return false;
    }

    size_t vx_size = (size_t)(nc / 8) * (n / 256) * 1152;
    size_t vy_size = (size_t)(n / 256) * sizeof(block_q8_K_kernel);
    size_t s_size  = nc * sizeof(float);

    gemv_config config;
    config.n  = n;
    config.bs = bs;
    config.nr = nr;
    config.nc = nc;

    if (ioctl(g_fpga_fd, FPGA_IOCTL_GEMV, &config) < 0) {
        return false;
    }

    if (pwrite(g_fpga_fd, vx, vx_size, FPGA_OFFSET_VX) < 0) {
        return false;
    }

    if (pwrite(g_fpga_fd, vy, vy_size, FPGA_OFFSET_VY) < 0) {
        return false;
    }

    if (pread(g_fpga_fd, s, s_size, FPGA_OFFSET_S) < 0) {
        return false;
    }

    return true;
}

static enum ggml_status ggml_backend_fpga_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_Q4_K && node->src[1]->type == GGML_TYPE_Q8_K && node->src[1]->ne[1] == 1) {
            int n  = (int)node->src[0]->ne[0];
            int nc = (int)node->src[0]->ne[1];
            if (!ggml_fpga_gemv_q4_K_8x8_q8_K(n, (float *)node->data, 0, node->src[0]->data, node->src[1]->data, 0, nc)) {
                return GGML_STATUS_FAILED;
            }
        } else {
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_fpga_interface = {
    /* .get_name                = */ ggml_backend_fpga_name,
    /* .free                    = */ ggml_backend_fpga_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_fpga_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_fpga_guid() {
    static ggml_guid guid = {0x12, 0x34, 0x56, 0x78, 0x90, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
    return &guid;
}

static const char * ggml_backend_fpga_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FPGA-DEV";
}

static const char * ggml_backend_fpga_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FPGA Matrix Accelerator";
}

static void ggml_backend_fpga_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 512LL * 1024 * 1024;
    *total = 512LL * 1024 * 1024;
}

static enum ggml_backend_dev_type ggml_backend_fpga_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_fpga_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_fpga_device_get_name(dev);
    props->description = ggml_backend_fpga_device_get_description(dev);
    props->memory_free = 512LL * 1024 * 1024;
    props->memory_total = 512LL * 1024 * 1024;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->caps = {
        /* .async_layer_norm = */ false,
        /* .async_softmax    = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events = */ false,
    };
}

ggml_backend_t ggml_backend_fpga_init(const char * device_path);
ggml_backend_buffer_type_t ggml_backend_fpga_buffer_type(const char * device_path);

static ggml_backend_t ggml_backend_fpga_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    return ggml_backend_fpga_init(params ? params : "/dev/fpga0");
}

static ggml_backend_buffer_type_t ggml_backend_fpga_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_fpga_buffer_type("/dev/fpga0");
}

static bool ggml_backend_fpga_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false; // Force CPU backend to handle graph, we intercept GEMV there
}

static bool ggml_backend_fpga_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return strcmp(ggml_backend_fpga_buffer_type_name(buft), "FPGA") == 0;
}

static struct ggml_backend_device_i ggml_backend_fpga_device_interface = {
    /* .get_name             = */ ggml_backend_fpga_device_get_name,
    /* .get_description      = */ ggml_backend_fpga_device_get_description,
    /* .get_memory           = */ ggml_backend_fpga_device_get_memory,
    /* .get_type             = */ ggml_backend_fpga_device_get_type,
    /* .get_props            = */ ggml_backend_fpga_device_get_props,
    /* .init_backend         = */ ggml_backend_fpga_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_fpga_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_fpga_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fpga_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_fpga_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FPGA";
}

static size_t ggml_backend_fpga_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_fpga_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(index);
    static struct ggml_backend_device fpga_dev = {
        /* .iface   = */ ggml_backend_fpga_device_interface,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &fpga_dev;
}

static struct ggml_backend_reg_i ggml_backend_fpga_reg_interface = {
    /* .get_name         = */ ggml_backend_fpga_reg_get_name,
    /* .get_device_count = */ ggml_backend_fpga_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fpga_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_fpga_buffer_type(const char * device_path) {
    GGML_UNUSED(device_path);
    static struct ggml_backend_buffer_type_i ggml_backend_fpga_buffer_type_interface_impl = {
        /* .get_name         = */ ggml_backend_fpga_buffer_type_name,
        /* .alloc_buffer     = */ ggml_backend_fpga_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_fpga_buffer_get_alignment,
        /* .get_max_size     = */ NULL,
        /* .get_alloc_size   = */ NULL,
        /* .is_host          = */ NULL,
    };
    static struct ggml_backend_buffer_type ggml_backend_fpga_buffer_type = {
        /* .iface    = */ ggml_backend_fpga_buffer_type_interface_impl,
        /* .device   = */ ggml_backend_fpga_reg_get_device(ggml_backend_fpga_reg(), 0),
        /* .context  = */ nullptr,
    };
    return &ggml_backend_fpga_buffer_type;
}

ggml_backend_t ggml_backend_fpga_init(const char * device_path) {
    ggml_backend_fpga_context * ctx = new ggml_backend_fpga_context(device_path);
    if (ctx->fd < 0) {
        delete ctx;
        return nullptr;
    }
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_fpga_guid(),
        /* .iface     = */ ggml_backend_fpga_interface,
        /* .device    = */ ggml_backend_fpga_reg_get_device(ggml_backend_fpga_reg(), 0),
        /* .context   = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_fpga(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_fpga_name;
}

ggml_backend_reg_t ggml_backend_fpga_reg(void) {
    static struct ggml_backend_reg ggml_backend_fpga_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fpga_reg_interface,
        /* .context     = */ nullptr,
    };
    return &ggml_backend_fpga_reg;
}
