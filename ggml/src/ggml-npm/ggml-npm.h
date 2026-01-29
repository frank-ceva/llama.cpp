#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_npm_init(void);
GGML_BACKEND_API bool ggml_backend_is_npm(ggml_backend_t backend);

// Backend registry
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_npm_reg(void);

#ifdef __cplusplus
}
#endif
