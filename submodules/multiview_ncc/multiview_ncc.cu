// ported/adapted (structural template only) from GaussianWrapping
// submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc/
// warp_patch_ncc.cu -- Torch tensor plumbing around the raw-pointer launcher
// in cuda_multiview_ncc/multiview_ncc_impl.cu. See multiview_ncc.h for why
// this is forward-only and projection-model-generic (PH + EQ).
#include <tuple>
#include "cuda_multiview_ncc/multiview_ncc_impl.h"
#include "multiview_ncc.h"

std::tuple<torch::Tensor, torch::Tensor>
MultiviewNCC(const torch::Tensor& depths,
             const torch::Tensor& normals,
             const torch::Tensor& uvs,
             const torch::Tensor& ray_dirs_r,
             const torch::Tensor& R,
             const torch::Tensor& T,
             const torch::Tensor& image_r,
             const torch::Tensor& image_n,
             const int render_model_n,
             const float fx_n, const float fy_n,
             const float cx_n, const float cy_n,
             const int patch_radius,
             const bool debug) {
    if (normals.ndimension() != 2 || normals.size(1) != 3) {
        AT_ERROR("normals must have dimensions (num_points, 3)");
    }
    if (uvs.ndimension() != 2 || uvs.size(1) != 2) {
        AT_ERROR("uvs must have dimensions (num_points, 2)");
    }
    if (ray_dirs_r.ndimension() != 3 || ray_dirs_r.size(2) != 3) {
        AT_ERROR("ray_dirs_r must have dimensions (H, W, 3)");
    }

    const int P     = depths.size(0);
    auto float_opts = depths.options().dtype(torch::kFloat32);

    torch::Tensor ncc   = torch::zeros({P}, float_opts);
    torch::Tensor valid = torch::zeros({P}, depths.options().dtype(torch::kBool));

    const int Hr = ray_dirs_r.size(0);
    const int Wr = ray_dirs_r.size(1);
    const int Hn = image_n.size(0);
    const int Wn = image_n.size(1);

    CHECK_CUDA(multiview_ncc_forward_launcher(
                   P,
                   depths.contiguous().data_ptr<float>(),
                   normals.contiguous().data_ptr<float>(),
                   uvs.contiguous().data_ptr<int>(),
                   ray_dirs_r.contiguous().data_ptr<float>(),
                   R.contiguous().data_ptr<float>(),
                   T.contiguous().data_ptr<float>(),
                   image_r.contiguous().data_ptr<float>(),
                   image_n.contiguous().data_ptr<float>(),
                   render_model_n,
                   fx_n, fy_n, cx_n, cy_n,
                   Hr, Wr, Hn, Wn,
                   patch_radius,
                   ncc.contiguous().data_ptr<float>(),
                   valid.contiguous().data_ptr<bool>()),
               debug);

    return std::make_tuple(ncc, valid);
}
