# ported/adapted from GaussianWrapping gaussian_wrapping/regularization/sdf/depth_fusion.py
# (transform_points_to_pixel_space) -- rewritten to project directly from
# view-space points using 3DGEER's own PH/EQ intrinsics instead of GW's
# full_proj_transform, so it is consistent with utils/ray_normals.py's
# get_ray_dirs_view() pixel convention (PINNED in Session C/E).
"""Batched view-space -> pixel-space projection for the two 3DGEER camera
models used by GUTWrap Path A (PH = pinhole, KB/EQ = fisheye equidistant).

The PH formula is the *exact* algebraic inverse of the ray formula used in
``utils/ray_normals.py::get_ray_dirs_view`` (render_model == 2 branch):

    ray = ((px + 0.5) / fx - W / (2 fx), (py + 0.5) / fy - H / (2 fy), 1)

=>  u = fx * (x / z) + W / 2 - 0.5
    v = fy * (y / z) + H / 2 - 0.5

The EQ formula follows the same "-0.5" pixel-center convention for
consistency, applied to the equidistant fisheye model
theta = atan2(sqrt(x^2+y^2), z), phi = atan2(y, x):

    u = principal_x + focal_x * theta * cos(phi) - 0.5
    v = principal_y + focal_y * theta * sin(phi) - 0.5

Note this is the *undistorted* equidistant model (no Kannala-Brandt radial
polynomial); the real per-pixel ``camera.raymap`` used by the CUDA
rasterizer/get_ray_dirs_view may include KB distortion coefficients baked in
at data-prep time (data/scnt/scnt_raymap.py). For X5 EQ data with
distortion_scaling=0 (see 3DGEERGW_EXECUTION.md D6 / Session F notes) the two
coincide; validation here is therefore a synthetic self-consistency check,
not a round-trip against a real distorted raymap (see tests/test_fisheye_proj.py).
"""
import math

import torch

_EQ_MAX_THETA = math.radians(89.0)


def project_view_to_pixel(camera, x_view: torch.Tensor):
    """Project batched view-space points to pixel coordinates.

    Args:
        camera: a scene.cameras.Camera (or MiniCam) instance. Must have
            ``render_model`` in {1 (KB/EQ), 2 (PH)}, plus the relevant
            intrinsics (focal_x/focal_y/principal_x/principal_y,
            image_width/image_height).
        x_view (torch.Tensor): (N, 3) points already in the camera's view
            space (see gaussian_wrapping/fields.py::world_to_view for the
            world -> view transform consistent with this camera convention).

    Returns:
        u (N,), v (N,) float pixel coordinates (pixel-center convention,
            i.e. pixel i occupies [i-0.5, i+0.5)); valid (N,) bool mask.
    """
    assert x_view.shape[-1] == 3
    x = x_view[..., 0]
    y = x_view[..., 1]
    z = x_view[..., 2]
    W = camera.image_width
    H = camera.image_height

    if camera.render_model == 2:  # PH (pinhole)
        valid_z = z > 1e-6
        safe_z = torch.where(valid_z, z, torch.ones_like(z))
        u = camera.focal_x * (x / safe_z) + W / 2.0 - 0.5
        v = camera.focal_y * (y / safe_z) + H / 2.0 - 0.5
        valid = (
            valid_z
            & (u >= -0.5) & (u <= W - 0.5)
            & (v >= -0.5) & (v <= H - 0.5)
        )
        return u, v, valid

    elif camera.render_model == 1:  # KB/EQ (equidistant fisheye, undistorted)
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(r, z)
        phi = torch.atan2(y, x)
        u = camera.principal_x + camera.focal_x * theta * torch.cos(phi) - 0.5
        v = camera.principal_y + camera.focal_y * theta * torch.sin(phi) - 0.5
        valid = (
            (theta < _EQ_MAX_THETA)
            & (u >= -0.5) & (u <= W - 0.5)
            & (v >= -0.5) & (v <= H - 0.5)
        )
        return u, v, valid

    else:
        raise NotImplementedError(
            f"project_view_to_pixel: render_model {camera.render_model} (BEAP) "
            "is not supported -- only PH (2) and KB/EQ (1) are implemented."
        )
