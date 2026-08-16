# Phase shift estimation alongside defocus (updated)

## Scope

- **torch-ctf**: `calculate_ctf_1d` and `calculate_ctf_2d` already accept `phase_shift` (degrees, 0–180). No changes there.
- **User-facing**: Phase shift estimation is optional and only active when the user explicitly requests it (e.g. `optimize_phase_shift=True`).

---

## Wrap-around: represent phase as unit vector (180° periodicity)

Phase shift θ is **periodic over 180°** (0° and 180° are equivalent). Use a unit-vector representation so wrapping is handled automatically:

1. **Represent phase as (u, v)**  
   For 180° periodicity, **double** the angle:
   - `u = cos(2θ)`, `v = sin(2θ)`  
   So 0° and 180° map to the same (u, v); wrap-around disappears in (u, v) space.

2. **Unit circle constraint**  
   Enforce `u² + v² = 1` with a small penalty in the loss:
   - Penalty: **λ (u² + v² − 1)²**  
   This keeps (u, v) on the unit circle and prevents amplitude drift during optimization.

3. **Recover phase**  
   After fitting u and v:
   - **θ = (1/2) · atan2(v, u)**  
   (in radians; convert to degrees for the CTF). This automatically handles wrapping.

---

## Quadratic model for phase shift (2D)

**No center.** Normalize patch positions so **x, y ∈ [-1, 1]** (e.g. `x = 2*pos_x - 1`, `y = 2*pos_y - 1`).

**Fit quadratics to the vector components** (not to θ directly):

- **u(x, y)** = C_u + a_u·x + b_u·y + c_u·x² + d_u·xy + e_u·y²  
- **v(x, y)** = C_v + a_v·x + b_v·y + c_v·x² + d_v·xy + e_v·y²  

So **12 coefficients** (two quadratics, 6 each). No explicit center; x and y are in [-1, 1].

**At each (x, y):**

1. Compute u(x,y), v(x,y).
2. Optionally normalize to unit circle: (u, v) := (u, v) / √(u² + v²), or rely on penalty λ(u² + v² − 1)².
3. Recover phase: **θ(x, y) = (1/2) · atan2(v, u)** (then convert to degrees for `calculate_ctf_2d`).

No clamping: wrap-around is handled by the (u, v) representation and recovery formula.

---

## 1. Data structures and models

**`src/torch_ctf_estimation/models/`**

- **CTF**: Already has `phase_shift_degrees`; fill from 1D/2D estimates when phase shift is optimized.
- Add **QuadraticPhaseShiftModel** (for 2D quadratic phase shift):
  - Two quadratics: coefficients (C_u, a_u, b_u, c_u, d_u, e_u) and (C_v, a_v, b_v, c_v, d_v, e_v).
  - Input: (x, y) normalized to [-1, 1].
  - Evaluation: compute u(x,y), v(x,y); then θ_rad = (1/2)*atan2(v, u); convert to degrees for CTF.
  - Optionally store or use unit-circle penalty weight λ.

**`src/torch_ctf_estimation/estimate_ctf_2d/`** (and `models/results_models.py` for Defocus2DResults)

- **Defocus2DResults**: Add optional fields when phase shift is estimated:
  - `phase_shift_degrees: Optional[float]` (e.g. mean or central value for reporting)
  - `phase_shift_model_type: Optional[Literal["grid", "quadratic"]]`
  - `phase_shift_model: Optional[Union[CubicCatmullRomGrid3d, QuadraticPhaseShiftModel]]` (or serializable equivalent)
  - `phase_shift_trace: Optional[list[float]]` for debugging if desired

---

## 2. 1D pipeline: grid search + refinement

**`src/torch_ctf_estimation/estimate_ctf_1d/`**

- **Grid search** (when `optimize_phase_shift=True`):
  - 5° grid over phase shift: 0, 5, …, 175.
  - Maximize ZNCC over (defocus, B, phase_shift) to get `best_phase_shift`.
  - Extend `_GridSearch1DResult` with `best_phase_shift`, `test_phase_shift_values`.
- **Refinement** (wrap-around handling):
  - Parameterize the single phase shift as **(u, v)** with u = cos(2θ), v = sin(2θ).
  - Optimize u and v (e.g. two scalars); add penalty **λ(u² + v² − 1)²** to the loss to keep (u,v) on the unit circle.
  - Recover phase: **θ = (1/2) · atan2(v, u)** (convert to degrees for `calculate_ctf_1d`). No clamping.
- **`estimate_ctf_1d`**: New kwargs `optimize_phase_shift`, `initial_phase_shift`; set `ctf_model.phase_shift_degrees` from estimated value.

---

## 3. 2D pipeline: grid vs quadratic (no linear)

**`src/torch_ctf_estimation/estimate_ctf_2d/`**

- **Shared**: `optimize_phase_shift`, `phase_shift_model: "grid" | "quadratic"`, `initial_phase_shift`. When False, pass `phase_shift=0`. In both grid and quadratic, use **(u, v)** representation and **θ = (1/2)·atan2(v, u)** so wrap-around is handled without clamping.
- **Grid model**: Store/optimize **(u, v)** per grid point (e.g. two 3D grids or one 3D grid with 2 channels). At each patch position, evaluate u and v from the grid; add penalty λ(u² + v² − 1)²; recover θ = (1/2)·atan2(v, u) in degrees; pass to `calculate_ctf_2d`.
- **Quadratic model**:
  - Normalize patch positions to **[-1, 1]** for x and y.
  - Fit u(x,y) and v(x,y) as two quadratics (12 coefficients total). Recover θ = (1/2)·atan2(v, u). Use penalty λ(u² + v² − 1)² to prevent amplitude drift.
- Both **`estimate_defocus_2d_grid`** and **`estimate_defocus_2d_linear`** (in `estimate_ctf_2d_utils.py`) accept phase-shift options and pass per-patch `phase_shift` (in degrees) into `calculate_ctf_2d`.

---

## 4. Top-level API and wiring

**`src/torch_ctf_estimation/estimate_ctf.py`**

- Add `optimize_phase_shift: bool = False`, `phase_shift_model: Literal["grid", "quadratic"] = "grid"`.
- Pass into `estimate_ctf_1d`, `_estimate_defocus_2d_at_1x1`, and `estimate_ctf_2d`.
- When `use_1d_defocus_for_spatial` is True and phase shift is estimated: use scalar from 1×1 or fit grid/quadratic to per-patch 1D phase values.

---

## 5. Tests and plotting

- Test 1D with `optimize_phase_shift=True`; 2D with `phase_shift_model="grid"` and `"quadratic"`; ensure `optimize_phase_shift=False` leaves phase at 0.
- Plotting already uses `ctf_model.phase_shift_degrees`; no change once 1D fills it.

---

## Summary: wrap-around and quadratic

- **180° periodicity**: Represent phase as **(u, v)** with u = cos(2θ), v = sin(2θ). Recover **θ = (1/2)·atan2(v, u)**. No clamping.
- **Unit circle**: Add penalty **λ(u² + v² − 1)²** to the loss to prevent amplitude drift.
- **2D quadratic**: x, y in **[-1, 1]**. Two quadratics: **u(x,y)** and **v(x,y)** (12 coefficients). θ(x,y) = (1/2)·atan2(v, u).
- **2D grid**: Store/optimize (u, v) per grid point; recover θ at each patch the same way.
- **1D refinement**: Single (u, v) with penalty; θ = (1/2)·atan2(v, u).
