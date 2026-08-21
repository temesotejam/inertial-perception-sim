from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _translation_nullspace(prev_b,cur_b,rotation):
    moved=rotation.apply(cur_b)
    # Epipolar constraint: prev^T [t]x R cur = 0, equivalently
    # t is orthogonal to cross(R cur, prev) for every correspondence.
    C=np.cross(moved,prev_b)
    _,_,vh=np.linalg.svd(C,full_matrices=False)
    t=vh[-1];n=np.linalg.norm(t)
    return (t/n if n>1e-12 else np.array([1.,0.,0.])),C


def epipolar_relative_rotation(prev_b,cur_b,rotation_prior,max_delta_deg=2.0):
    """Estimate relative rotation while marginalizing monocular translation direction.

    The unknown translation scale is irrelevant to the essential constraint. For
    each candidate rotation the best translation direction is the null vector of
    the epipolar cross-product matrix, so only a 3-D rotation correction around
    the inertial prior must be optimized. The prior resolves the otherwise weak
    rotation/translation ambiguity but does not provide the final image residual.
    """
    prev_b=np.asarray(prev_b,float);cur_b=np.asarray(cur_b,float)
    if len(prev_b)<5 or len(cur_b)!=len(prev_b) or rotation_prior is None:return None
    limit=np.radians(float(max_delta_deg))
    prior_scale=np.radians(.9)

    def rotation_of(x):
        return rotation_prior*Rotation.from_rotvec(np.asarray(x,float))

    def fun(x):
        rot=rotation_of(x);t,C=_translation_nullspace(prev_b,cur_b,rot)
        epi=C@t
        # A weak prior stabilizes low-parallax/near-degenerate frames. It is
        # deliberately much looser than normal frame-to-frame IMU error.
        reg=.08*np.asarray(x,float)/prior_scale
        return np.r_[epi,reg]

    try:
        sol=least_squares(fun,np.zeros(3),bounds=(-limit,limit),loss='soft_l1',f_scale=.002,max_nfev=60)
    except Exception:return None
    rot=rotation_of(sol.x);t,C=_translation_nullspace(prev_b,cur_b,rot);epi=C@t
    epi_rms_deg=float(np.degrees(np.sqrt(np.mean(epi**2))))
    correction_deg=float(np.degrees(np.linalg.norm(sol.x)))
    return {'rotation':rot,'translation_direction_previous':t,'epipolar_rms_deg':epi_rms_deg,'prior_correction_deg':correction_deg,'success':bool(sol.success)}
