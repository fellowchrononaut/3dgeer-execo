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

The KB/EQ formula applies the equidistant fisheye model
theta = atan2(sqrt(x^2+y^2), z), phi = atan2(y, x), plus the same forward
Kannala-Brandt radial polynomial used to build the per-pixel
``camera.raymap`` at data-prep time (data/scnt/scnt_raymap.py::compute_error_map):

    theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    u = principal_x + focal_x * theta_d * cos(phi)
    v = principal_y + focal_y * theta_d * sin(phi)

using ``camera.distortion_coeffs`` (k1..k4). For X5 EQ data with
distortion_scaling=0 (see 3DGEERGW_EXECUTION.md D6 / Session F notes)
``distortion_coeffs`` is all zeros, so theta_d reduces to theta and this
collapses back to the plain undistorted equidistant model -- the fix is a
strict generalization, not a dataset-specific branch.

Note this branch intentionally has *no* "-0.5" pixel-center shift, unlike
the PH branch above. data/scnt/scnt_raymap.py builds the per-pixel raymap
from raw integer pixel indices (``np.meshgrid(np.arange(W), np.arange(H))``,
no +0.5), and get_ray_dirs_view's render_model==1 path reads that raymap
directly with no offset either -- so this formula must match that same
zero-offset convention to round-trip against the real, rendered rays (see
tests/test_fisheye_proj.py check4, which caught a spurious ~0.71px bias
before this was removed).
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

    elif camera.render_model == 1:  # KB/EQ (equidistant fisheye)
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(r, z)
        phi = torch.atan2(y, x)

        dc = getattr(camera, "distortion_coeffs", None)
        if dc is not None:
            dc = dc.to(device=x_view.device, dtype=x_view.dtype)
            k1, k2, k3, k4 = dc[0], dc[1], dc[2], dc[3]
            theta2 = theta * theta
            theta_d = theta * (1 + k1 * theta2 + k2 * theta2**2
                                + k3 * theta2**3 + k4 * theta2**4)
        else:
            theta_d = theta

        u = camera.principal_x + camera.focal_x * theta_d * torch.cos(phi)
        v = camera.principal_y + camera.focal_y * theta_d * torch.sin(phi)
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
