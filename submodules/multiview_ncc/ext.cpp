#include <torch/extension.h>
#include "multiview_ncc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multiview_ncc_forward", &MultiviewNCC);
}
