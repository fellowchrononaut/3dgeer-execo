# multiview_ncc — analytic backward derivation (Phase 2)

Single source of truth for the math implemented in
`cuda_multiview_ncc/multiview_ncc_impl.cu` (forward kernel +
`multiview_ncc_backward_kernel<T>`). Companion documents:
`GUTWrap_Discussion/3DGEERGW_EXECUTION.md` Session F2 (decision history),
`utils/multiview.py` (autograd wiring + pure-torch oracle),
`tests/test_multiview.py` (all numbers quoted below are reproduced by it).

Notation: one *query point* `i` = one reference-image pixel `(x_c, y_c)`
(= `uvs[i]`) with scalar `depth` and ref-view-space `normal` **n** (need not
be unit). One *patch offset* `k` = integer pixel `(x_c+du, y_c+dv)`,
`du,dv ∈ [-radius, radius]`, `M = (2·radius+1)²` offsets total
(production `radius = 3`, `M = 49`).

---

## 1. Forward chain (exact formulas as implemented)

Implemented in `multiview_ncc_forward_kernel` (float32, compiled with
`--use_fast_math`). The pure-torch oracle
(`utils/multiview.py::ncc_reference_oracle`) implements the identical math
with `grid_sample`/tensor ops.

**Stage F1 — ray lookup.** Per offset `k`, the reference camera's view-space
unit ray `d_k = ray_dirs_r[y_k·Wr + x_k]` (precomputed by
`utils/ray_normals.py::get_ray_dirs_view`, passed in as an `(Hr,Wr,3)`
buffer). The center ray is `d_c = ray_dirs_r[y_c·Wr + x_c]`.

**Stage F2 — tangent-plane intersection.** The query point anchors a plane
through `P0 = depth·d_c` with normal **n**. The offset ray hits it at

```
a   = n·d_c                (shared by all offsets)
b_k = n·d_k                ("denom" in the kernel)
t_k = depth·a / b_k        (kernel computes N_dot_P0 = n·P0 = depth·a first)
X_k = t_k · d_k            (ref view space)
```

Guards: if `|b_k| < 1e-8`, `b_k` is replaced by `copysign(1e-8, b_k)`
(`sdenom`) and the offset is flagged `bad`; `t_k ≤ 0` (behind the camera)
also flags `bad`. `bad` feeds the validity mask (§4), **not** a skip — the
sample is still taken (at a clamped location) so the arithmetic never reads
out of bounds.

**Stage F3 — rigid transform to the neighbor view.**

```
Xn_k = R @ X_k + T         (R row-major 3x3, T (3,), ref-view → neighbor-view;
                            from utils/multiview.py::_ref_to_neighbor_RT)
```

**Stage F4 — projection to neighbor pixels** (mirrors
`gaussian_wrapping/fisheye_proj.py::project_view_to_pixel`, pixel-center
convention: pixel `i` is centered at continuous coordinate `i`).

PH (`render_model_n == 2`), with guard `z > 1e-6` (else `safe_z = 1`,
invalid):

```
u = fx·(x/z) + W/2 − 0.5
v = fy·(y/z) + H/2 − 0.5
valid: z > 1e-6  ∧  u ∈ [−0.5, W−0.5]  ∧  v ∈ [−0.5, H−0.5]
```

EQ (`render_model_n == 1`, undistorted equidistant):

```
r = √(x²+y²),  θ = atan2(r, z),  φ = atan2(y, x)
u = cx + fx·θ·cos φ − 0.5
v = cy + fy·θ·sin φ − 0.5
valid: θ < 89°  ∧  u ∈ [−0.5, W−0.5]  ∧  v ∈ [−0.5, H−0.5]
```

(The backward kernel computes the *same* u,v via the exact identity
`cos φ = x/r`, `sin φ = y/r` — `project_eq_identity` — which avoids the
`atan2(y,x)/cos/sin` calls; the r→0 limit agrees since `θ·(x/r) → 0`.)

**Stage F5 — sampling.** `c_n_k = bilinear(image_n, u_k, v_k)` with all four
corner indices clamped to `[0,W−1]×[0,H−1]` (== `grid_sample`
`padding_mode="border"`). `c_r_k = image_r[y_k·Wr + x_k]` — an exact integer
pixel read (no interpolation; ref offsets are integer by construction).

**Stage F6 — NCC.** With `S_r = Σc_r`, `S_n = Σc_n`, `S_rr = Σc_r²`,
`S_nn = Σc_n²`, `S_rn = Σc_r·c_n` over the M offsets:

```
cross = S_rn − S_r·S_n/M
var_r = S_rr − S_r²/M
var_n = S_nn − S_n²/M
NCC   = cross² / (var_r·var_n + 1e-8)
valid_patch = all_inside ∧ var_r > 5e-6 ∧ var_n > 5e-6
ncc[i] = valid_patch ? NCC : 0
```

(`all_inside` = every offset had `proj_valid ∧ ¬bad`. This squared-NCC form
and the epsilons are ported verbatim from GGGS's `warp_patch_ncc`; NCC here
is cross-correlation *squared* over the two variances, range ≈ [0,1].)

---

## 2. Backward chain (closed-form Jacobians, same notation)

Implemented in `multiview_ncc_backward_kernel<T>` (T = double by default,
see §5). Only `depths` and `normals` receive gradients — `c_r`, `var_r`,
the rays, R/T and the images are constants w.r.t. them. The kernel
recomputes the forward reduction per point (Pass A, which also stashes
per-offset `u_k, v_k, c_n_k` and, for EQ, `r_k, θ_k`), then accumulates the
chain per offset (Pass B).

**(a) NCC w.r.t. one warped sample, with the patch-statistics coupling.**
Only `c_n_k` depends on (depth, n). From F6, using
`∂cross/∂c_n_k = c_r_k − S_r/M` and `∂var_n/∂c_n_k = 2(c_n_k − S_n/M)`:

```
den = var_r·var_n + 1e-8
A   = ∂NCC/∂cross · … = 2·cross/den
B   = −2·∂NCC/∂var_n = 2·cross²·var_r/den²
∂NCC/∂c_n_k = A·(c_r_k − S_r/M) − B·(c_n_k − S_n/M)
```

The `−S_r/M`/`−S_n/M` terms are the coupling of every sample to the patch
means — dropping them is the classic NCC-backward mistake; they are exact
here because `∂S_n/∂c_n_k = 1` folds into each offset's own derivative.

**(b) Bilinear gradient.** With `x0 = ⌊u⌋`, `y0 = ⌊v⌋`, weights
`wx1 = u−x0`, `wy1 = v−y0` (corner indices clamped as in F5):

```
∂c/∂u = (c01−c00)·wy0 + (c11−c10)·wy1
∂c/∂v = (c10−c00)·wx0 + (c11−c01)·wx1
```

Piecewise constant-in-`u` / linear-in-`v` (and vice versa) inside each pixel
cell; **discontinuous across integer u,v** — see §4/§5 for how that shows up
in validation.

**(c) Projection Jacobians ∂(u,v)/∂(x,y,z) at `Xn_k`.**

PH (`ph_projection_jacobian`):

```
∂u/∂x = fx/z     ∂u/∂y = 0        ∂u/∂z = −fx·x/z²
∂v/∂x = 0        ∂v/∂y = fy/z     ∂v/∂z = −fy·y/z²
```

EQ (`eq_projection_jacobian`), using `cφ = x/r`, `sφ = y/r`, `R2 = r²+z²`:

```
∂u/∂x = fx·( z·x·cφ/(r·R2) + θ·y²/r³ )
∂u/∂y = fx·( z·y·cφ/(r·R2) − θ·x·y/r³ )
∂u/∂z = −fx·r·cφ/R2
∂v/∂x = fy·( z·x·sφ/(r·R2) − θ·x·y/r³ )
∂v/∂y = fy·( z·y·sφ/(r·R2) + θ·x²/r³ )
∂v/∂z = −fy·r·sφ/R2
```

Derived from `∂θ/∂(x,y) = (z/R2)·(x,y)/r`, `∂θ/∂z = −r/R2`,
`∂φ/∂(x,y) = (−y, x)/r²`, `∂(cos φ)/∂x = y²/r³`, etc. In code the `θ·…/r³`
terms are written `θ·sφ·k2·y` with `k2 = 1/r²` (identical via `sφ = y/r`).

**ρ→0 guard:** as `r → 0`, `θ → 0` at the same rate the `1/r`-type factors
grow, so the true Jacobian is bounded (direction-dependent). The code floors
the offending denominators (`r_safe = max(r, 1e-9)`,
`rho2_safe = max(r², 1e-18)`, `R2_safe = max(R2, 1e-18)` in double) instead
of special-casing the limit. This region is *exercised* by the tests (patch
centers project near the neighbor's optical axis in the EQ fixture) and
matches the f64 oracle to machine precision.

**(d) Chain to depth and normal.** `Xn_k = t_k·(R d_k) + T`, so
`∂Xn_k/∂t_k = R d_k =: d'_k` exactly (t is the only free scalar):

```
∂u/∂t = J_u · d'_k        ∂v/∂t = J_v · d'_k       (directional derivatives)
∂c_n_k/∂t_k = ∂c/∂u·∂u/∂t + ∂c/∂v·∂v/∂t
```

Ray–plane intersection `t_k = depth·a/b_k`, quotient rule:

```
∂t_k/∂depth = a / b_k
∂t_k/∂n     = depth·(b_k·d_c − a·d_k) / b_k²        (3-vector)
```

(uses `∂a/∂n = d_c`, `∂b_k/∂n = d_k`; `b_k` is the guarded `sdenom` in code.)

**Total**, with upstream `g_i = ∂L/∂ncc_i`:

```
∂L/∂depth_i = g_i · Σ_k  ∂NCC/∂c_n_k · ∂c_n_k/∂t_k · ∂t_k/∂depth
∂L/∂n_i     = g_i · Σ_k  ∂NCC/∂c_n_k · ∂c_n_k/∂t_k · ∂t_k/∂n
```

---

## 3. Sign conventions — what was verified empirically vs derived

Per the project convention (empirical sign verification over
trust-the-derivation — see the PAM analytic-gradient session log), **every
sign above is pinned end-to-end** by
`tests/test_multiview.py::check_analytic_backward_vs_oracle`: the kernel's
gradients equal the float64 oracle's *autograd* gradients (torch derives
those signs mechanically) to machine precision, for PH and EQ, on 256
moderately-off query points each. No individual sign relies on the hand
derivation alone. Specific spots that were verified rather than trusted:

- `B`'s sign (the `−∂NCC/∂var_n` term flips twice: once from the quotient
  rule, once from folding `−` into `B`'s definition) — pinned by the oracle
  check.
- `∂t/∂n`'s antisymmetric `b·d_c − a·d_k` order — pinned by the oracle check
  (swapping it flips the normal gradient for off-center offsets only, which
  the 49-offset sum would partially mask; the machine-precision agreement
  rules it out).
- EQ Jacobian `∂u/∂y` / `∂v/∂x` minus signs — pinned by the oracle check on
  the EQ fixture.
- Upstream orientation (training minimizes `1−ncc`, so `∂L/∂ncc = −weight`
  arrives via autograd) — unchanged from Phase 1; the FD cross-check
  (`check_analytic_vs_fd_backward`) confirms the analytic gradients point
  the same way as the already-training-validated FD gradients 80–86% of the
  time (the disagreements are FD noise, §5).

---

## 4. Non-differentiable boundaries and gradient masking

The loss is **piecewise smooth**. Boundaries, and what the backward does:

1. **Patch bbox** (ref patch would leave the reference image): forward
   writes `ncc = 0, valid = false` before any read; backward writes zero
   gradients and returns. Correct: the value is constant (0) on that side.
2. **`valid_patch = false`** (any offset `bad` / projection invalid / either
   variance ≤ 5e-6): forward gates `ncc` to exactly 0, so the loss is
   *disconnected* there; backward mirrors the same recomputed gate and
   returns exact zeros. Not smoothed over — training's own mask
   (`ncc_valid`) already excludes these points from the loss.
3. **Bilinear cell edges** (integer `u_n` or `v_n`): `c_n` is C⁰ but its
   derivative jumps. Both the kernel and the oracle compute the exact
   one-sided derivative of whichever cell the point lands in; with double
   internals both land in the same cell (agreement to f64 ulp), so nothing
   needs masking in the precise mode. In any float32 implementation
   (including the f32 oracle), ~1e-3 px coordinate noise makes the two sides
   disagree at points that straddle an edge — that is measurement noise of
   f32, not a defect of either implementation (§5).
4. **EQ θ-cone (89°) and image-border validity**: crossing flips
   `proj_valid` → case 2.
5. **`b_k → 0`** (ray parallel to the tangent plane): `t_k → ±∞`; the
   `1e-8` floor keeps arithmetic finite and the `t ≤ 0`/projection checks
   turn the offset `bad` → case 2.

**Out-of-bounds audit (2026-07-07, after the sessF2_bundle30k crash):**
(a) the bilinear fetch at `u ∈ [W−1, W)` (and every other u,v, including
wildly invalid projections — samples are taken even when `proj_valid` is
false) clamps **all four** corner indices to `[0,W−1]×[0,H−1]` *before* any
read; GPU float→int conversion saturates rather than wrapping, and the
clamp then bounds it. (b) reference-image patch reads are gated by the bbox
early-return *before* any memory access. (c) no index derived from
valid-masked values is ever used unguarded. **No OOB path was found for
well-formed inputs.** Two defensive fixes were added anyway: the bbox check
was rewritten in an overflow-proof form (`uvc.x < radius || uvc.x >
Wr−1−radius`, arithmetic on trusted image dims only — the old
`uvc.x + radius` could wrap for garbage int32 uvs and slip past the check),
and the torch wrapper now rejects `depths/normals/uvs/grad_ncc` row-count
mismatches (a short `uvs` would otherwise read past its end in-kernel).

---

## 5. Validation results (tests/test_multiview.py, RTX 5090, 2026-07-07)

**Precision design.** The backward kernel is templated on its internal
scalar `T`:

- `T = double` (**precise**, the production default for the analytic mode):
  computes the exact derivative of the shared real-valued function. fp64
  runs at 1/64 rate on consumer GPUs, so this lands at ≈ FD speed.
- `T = float` (**fast**, `MULTIVIEW_NCC_ANALYTIC_PRECISE=0`): ~3x faster
  than FD end-to-end, with the noise floor *inherent to any f32
  implementation of this gradient* — established by comparing the f32
  reference oracle itself against a f64 oracle and observing the same error
  tail (cancellation in the `A(…)−B(…)` accumulation + fast-math u_n noise
  flipping bilinear cells). An all-f32 backward is **not** less correct than
  the f32 oracle; the tail is measurement noise shared by both.

**(a) Analytic (precise) vs float64 oracle autograd** — 256 points/model,
±40% depth, ±0.3 normal perturbations, noise floor `1e-3·RMS(grad)`:

| | grad_depths max rel err | grad_normals max rel err | masked |
|---|---|---|---|
| PH | 0.0 (bit-equal after f32 rounding) | 0.0 | 0 straddles, 0 valid-disagreements |
| EQ | 9.0e-8 | 0.0 (max abs diff below floor 1.8e-12) | 0 straddles, 0 valid-disagreements |

Masked fraction: **zero** — with double internals no bilinear-straddle or
validity-boundary masking was needed. Points below the noise floor
(~4% of depths entries, ~13% of normals entries — legitimately tiny true
gradients near NCC plateaus) are excluded from *relative* stats but still
agree absolutely (max abs diff ≤ 1.8e-12).

**Analytic (fast/f32) vs float64 oracle** (report-only): sign agreement
**1.000** on every model/quantity above the floor; rel err p50 ≈ 5e-4–1e-3,
p99 ≈ 4–7e-2, worst 0.10–0.42 at small-|grad|/straddle points — the
documented f32 noise floor.

**(b) Analytic vs calibrated FD backward** (eps = 3e-2, the Phase-1
production path, both run through `EQWarpPatchNCC` end-to-end; forward
outputs asserted **bit-identical** across modes):

| | depths sign agree | normals sign agree | mean abs diff (vs analytic scale) |
|---|---|---|---|
| PH | 0.851 | 0.800 | 2.8e-3 / 3.5e-2 (depths), 3.0e-3 / 5.6e-3 (normals) |
| EQ | 0.863 | 0.799 | 3.0e-3 / 4.1e-2 (depths), 3.0e-3 / 5.3e-3 (normals) |

This 80–86% envelope equals the Phase-1 FD-vs-oracle measurement (0.79–0.88)
— i.e. the disagreement is entirely the FD arm's noise, as expected.

**(c) Timing** (fwd+backward, avg over 20–30 runs):

| points | model | FD (9 fwd calls) | analytic64 | analytic32 |
|---|---|---|---|---|
| 10k | PH | 0.306 ms | 0.608 ms (0.50x) | 0.099 ms (**3.1x**) |
| 10k | EQ | 0.446 ms | 0.839 ms (0.53x) | 0.185 ms (**2.4x**) |
| 100k | PH | 1.998 ms | 1.937 ms (1.03x) | 0.656 ms (**3.1x**) |
| 100k | EQ | 1.859 ms | 2.591 ms (0.72x) | 0.619 ms (**3.0x**) |

Backward-path-only (subtracting one forward call): analytic32 ≈ **4x** over
FD. The originally expected 5–8x assumed an f32 backward ≈ 1 forward-call
cost; the measured f32 kernel is ≈ 2 forward-equivalents (it recomputes the
forward reduction and runs the Jacobian chain), hence ~3–4x, not 5–8x.
analytic64 buys exactness at FD-parity speed — its value is gradient
*quality* (exact vs FD's 80% sign agreement), not speed.

**Wiring**: `utils/multiview.py::MULTIVIEW_NCC_BACKWARD` = `"fd"` (default,
unchanged production behavior) | `"analytic"`; env-overridable. Analytic
precision: `MULTIVIEW_NCC_ANALYTIC_PRECISE` = `"1"` (default, double) |
`"0"` (float). Flipping production to the analytic backward is the one-line
env change `MULTIVIEW_NCC_BACKWARD=analytic` (no code edit).

---

## 6. Extending to KB (Kannala–Brandt) distortion — P4 recipe

When Metashape calibration of the X5 shows non-trivial distortion, the EQ
model gains the radial polynomial

```
θ_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
```

**Forward change (2 places, identical edit):** in
`project_to_neighbor_pixel` (float, forward kernel) and
`project_eq_identity` (templated, backward Pass A), replace `θ` by `θ_d` in
the pixel formulas only:

```
u = cx + fx·θ_d·cos φ − 0.5,   v = cy + fy·θ_d·sin φ − 0.5
```

(θ itself — the validity cone check — stays undistorted. Mirror the same
edit in `gaussian_wrapping/fisheye_proj.py::project_view_to_pixel` and the
oracle inherits it automatically, keeping oracle/kernel comparable.)

**Backward change (1 factor):** the EQ Jacobian in §2(c) decomposes as
θ-terms (the `z·…/(r·R2)` and `−r/R2` entries, which came from `∂θ/∂(x,y,z)`)
and φ-terms (the `θ·…/r³` entries). Under distortion:

- every occurrence of `∂θ/∂·` picks up the scalar chain factor
  `dθ_d/dθ = 1 + 3k1·θ² + 5k2·θ⁴ + 7k3·θ⁶ + 9k4·θ⁸` — i.e. in
  `eq_projection_jacobian`, multiply `k1` (the code variable, = ∂θ/∂r / r)
  and the `−r/R2` (= ∂θ/∂z) factor by `dθ_d/dθ`;
- every *bare* `θ` multiplying a φ-derivative term becomes `θ_d` — i.e. the
  code's `theta` in the `theta·sφ·k2·…` / `theta·cφ·k2·…` terms.

Concretely: pass `θ` and `r` from the Pass-A stash (already stashed),
compute `θ_d` and `dθ_d/dθ` once per offset (Horner, 4 fma), and use
`(k1·dθ_d/dθ)`, `(−r/R2·dθ_d/dθ)`, and `θ_d` in place of `k1`, `−r/R2`, `θ`.
No other stage of the chain changes (the bilinear, NCC, and ray-plane parts
are projection-agnostic). Re-run
`tests/test_multiview.py::check_analytic_backward_vs_oracle` with a KB
oracle to re-pin signs — the f64 comparison will catch any slip at machine
precision.
