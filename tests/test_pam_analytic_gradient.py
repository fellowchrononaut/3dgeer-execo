"""Session E3 PAM-noise-fix verification (GUTWrap Path A) -- run inside the
`geer` container:
    docker exec geer bash -lc "cd /home && python -m tests.test_pam_analytic_gradient"

Verifies, with SYNTHETIC Gaussians only (no real scene/checkpoint needed),
the sign convention of GW's ported analytic Newton-step gradient
(`gaussian_wrapping.pam_utils.get_vector_field_quantities_aux` /
`GaussianVectorField`) against OUR field's zero-centered convention
(field > 0 = empty/vacant, field < 0 = occupied, surface at field == 0).

This matters because `pam_extraction.py::_gradient_descent_refinement`'s
Newton step (`alpha = (target - field(x)) / |grad|^2`, `x <- x + 0.5*alpha*
grad`) was derived/verified for the *numerical* gradient of our own field_fn
(literally `grad(field_fn)`, pointing toward increasing field = increasing
vacancy = outward). Whether GW's independently-derived analytic vector field
points the same way is NOT obvious from the formula alone (the half-space
indicator's clipping behavior makes a hand-derivation easy to get backwards)
-- this test checks it directly instead of trusting the derivation.

Two checks:
  1. Direct sign check: a small flat synthetic Gaussian "surface patch" at
     x=0 with outward normal (1,0,0); the raw analytic gradient at a query
     point just outside (x=+0.15) must have a POSITIVE x-component (pointing
     further outward, away from the patch).
  2. End-to-end Newton-step check: using a hand-defined synthetic
     field_fn(pts) = pts[:,0] (a local-linear stand-in for "our real field"
     near this flat patch -- positive x = empty, negative x = occupied,
     zero-crossing at x=0, matching our real convention) together with the
     UNMODIFIED analytic grad_fn, one `_gradient_descent_refinement` step
     must move a point starting at x=+0.15 CLOSER to x=0 (x decreases), and
     a point starting at x=-0.05 (near a second synthetic patch at x=-0.15,
     also outward normal (1,0,0)) must likewise move CLOSER to x=0 (x
     increases).

If either check fails with the raw (unflipped) sign, this test will flip the
sign (`grad_fn = lambda pts: -vector_field.gradient(...)`) and re-verify --
whichever sign passes is the one documented in pam_utils.py's
`GaussianVectorField` docstring and wired into pam_extraction.py's
`--gradient_mode analytic`.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_wrapping.pam_utils import get_vector_field_quantities_aux  # noqa: E402
from gaussian_wrapping.pam_extraction import _gradient_descent_refinement  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_flat_patch(mu_x: float, normal_x: float, scale: float = 0.05, opacity: float = 0.9,
                      grid: tuple = (-0.1, 0.0, 0.1)) -> dict:
    """A small 3x3 grid of tightly-spaced, isotropic Gaussians on the plane
    x=mu_x (spanning y,z in `grid`), all sharing the same outward normal
    (normal_x, 0, 0), scale, and opacity -- a synthetic stand-in for a small
    flat surface patch. Returns contiguous (k,*) tensors (k=9)."""
    ys, zs = np.meshgrid(grid, grid, indexing="ij")
    ys, zs = ys.flatten(), zs.flatten()
    k = ys.shape[0]
    means = torch.tensor(
        np.stack([np.full(k, mu_x), ys, zs], axis=1), dtype=torch.float32, device=DEVICE)
    normals = torch.tensor([[normal_x, 0.0, 0.0]] * k, dtype=torch.float32, device=DEVICE)
    scales = torch.full((k, 3), scale, dtype=torch.float32, device=DEVICE)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * k, dtype=torch.float32, device=DEVICE)
    opacities = torch.full((k, 1), opacity, dtype=torch.float32, device=DEVICE)
    return dict(means=means, normals=normals, scales=scales, rotations=rotations, opacities=opacities)


def _broadcast_grad_fn(patch: dict):
    """Builds a grad_fn(pts) -> (N,3) that evaluates get_vector_field_quantities_aux
    against the SAME fixed patch of Gaussians for every query point (bypasses
    the real KD-tree -- we supply the "neighbors" directly, as suggested by
    the task spec for this synthetic test)."""
    def grad_fn(pts: torch.Tensor) -> torch.Tensor:
        n = pts.shape[0]
        k = patch["means"].shape[0]
        gm = patch["means"].unsqueeze(0).repeat(n, 1, 1)
        gn = patch["normals"].unsqueeze(0).repeat(n, 1, 1)
        gs = patch["scales"].unsqueeze(0).repeat(n, 1, 1)
        gr = patch["rotations"].unsqueeze(0).repeat(n, 1, 1)
        go = patch["opacities"].unsqueeze(0).repeat(n, 1, 1)
        return get_vector_field_quantities_aux(pts, gm, gn, gs, gr, go)
    return grad_fn


def test_direct_sign_check():
    patch = _make_flat_patch(mu_x=0.0, normal_x=1.0)
    query = torch.tensor([[0.15, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    grad = get_vector_field_quantities_aux(
        query,
        patch["means"].unsqueeze(0), patch["normals"].unsqueeze(0),
        patch["scales"].unsqueeze(0), patch["rotations"].unsqueeze(0), patch["opacities"].unsqueeze(0),
    )
    gx = grad[0, 0].item()
    print(f"[check 1] raw analytic gradient at query x=+0.15 (patch at x=0, outward +x): "
          f"grad = {grad[0].tolist()}")
    assert gx > 0, f"expected positive x-component (points further outward), got {gx}"
    print("PASS check 1: raw analytic gradient x-component is positive (outward), no sign flip needed.")
    return True  # sign is correct as-is


def _field_fn(pts: torch.Tensor) -> torch.Tensor:
    """Local-linear stand-in for OUR real field near a flat patch at x=0:
    positive x = empty, negative x = occupied, zero-crossing at x=0 -- exactly
    our real convention (field>0 empty, field<0 occupied, surface at 0)."""
    return pts[:, 0].clone()


def _run_refinement_step(start_x: float, patch: dict, sign: float):
    grad_fn_raw = _broadcast_grad_fn(patch)
    grad_fn = (lambda pts: sign * grad_fn_raw(pts))
    points = np.array([[start_x, 0.0, 0.0]], dtype=np.float32)
    refined, _, grad_norms = _gradient_descent_refinement(
        points, _field_fn, grad_fn, n_steps=1, min_grad_norm=1e-8, target=0.0, device=DEVICE)
    return refined[0, 0], grad_norms[0].item()


def test_refinement_step_outside_point(sign: float):
    patch = _make_flat_patch(mu_x=0.0, normal_x=1.0)
    new_x, gnorm = _run_refinement_step(0.15, patch, sign)
    print(f"[check 2a] sign={sign:+.0f}: point at x=+0.15 (vacant side) -> x={new_x:.6f} "
          f"(|grad|={gnorm:.4f})")
    return new_x < 0.15


def test_refinement_step_inside_point(sign: float):
    # Second patch at x=-0.15, SAME outward-normal orientation (+x) -- the
    # query point at x=-0.05 sits "outward" (on the +x side) relative to it.
    patch2 = _make_flat_patch(mu_x=-0.15, normal_x=1.0)
    new_x, gnorm = _run_refinement_step(-0.05, patch2, sign)
    print(f"[check 2b] sign={sign:+.0f}: point at x=-0.05 (occupied side, near patch@x=-0.15) "
          f"-> x={new_x:.6f} (|grad|={gnorm:.4f})")
    return new_x > -0.05


def main():
    sign_ok_raw = test_direct_sign_check()
    sign = 1.0 if sign_ok_raw else -1.0

    ok_a = test_refinement_step_outside_point(sign)
    ok_b = test_refinement_step_inside_point(sign)

    if not (ok_a and ok_b):
        print(f"[pam-sign-test] sign={sign:+.0f} failed the end-to-end refinement check "
              f"(outside_ok={ok_a}, inside_ok={ok_b}) -- flipping and re-verifying.")
        sign = -sign
        ok_a = test_refinement_step_outside_point(sign)
        ok_b = test_refinement_step_inside_point(sign)
        assert ok_a and ok_b, (
            f"Neither sign (+1 nor -1) produces a correct Newton-step direction "
            f"(outside_ok={ok_a}, inside_ok={ok_b}) -- something else is wrong "
            f"with get_vector_field_quantities_aux, not just a sign flip."
        )

    print(f"\n[pam-sign-test] VERIFIED SIGN = {sign:+.0f} "
          f"({'no flip' if sign > 0 else 'FLIP REQUIRED'} relative to raw "
          "get_vector_field_quantities_aux / GaussianVectorField.gradient output).")
    assert sign > 0, (
        "Sign flip is required but pam_extraction.py currently wires "
        "GaussianVectorField.gradient() unmodified -- update the wiring "
        "(and pam_utils.py's GaussianVectorField docstring) to negate it."
    )
    print("\nAll PAM analytic-gradient sign-convention checks PASSED.")


if __name__ == "__main__":
    main()
