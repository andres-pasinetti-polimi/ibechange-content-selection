"""The working model of the composition and its posterior.

implementation_plan_composition_2026-11-04.md, Phase 2, and rct/design/rct_design_record_2026-10-27.md,
Sections 1.2 and 12. The posterior of the effects is obtained from the Bayesian regression of the daily
outcome r on

    r = alpha_i + psi' alpha_0 + (g(x) - E_p[g]) ' theta + (lag sums) ' beta + noise,

where x = (R, Q) is the composition of the day, g(x) its seven treatment terms
(composition.TERM_NAMES: the increments of the first and second recommendation, of the first to
fourth reminder, and R x Q), E_p[g] their expectation under the probabilities of the day, so that
each term is centred (action-centred least squares), and the lag sums are the sums over the
design.LAG_DAYS preceding days of R - E[R] and of Q - E[Q], each centred under the probabilities of
its own day. alpha_i is a per-participant level, integrated out block by block, so the cost grows
linearly with the number of participants. No moderator multiplies the treatment terms and there is
no per-participant random effect on them (design record, Section 1.2; moderators are open, Section 2
of the plan). The noise variance and the variance of alpha_i are fixed.

beta estimates the effect of one recommendation and of one reminder sent today on the mark of one
later day, so that a burden of extra notifications on the following days is estimated from the
marks and not set by hand.
"""

from dataclasses import asdict, dataclass

import numpy as np

from cs_rct import composition, design

NAMES = composition.TERM_NAMES + composition.LAG_NAMES


@dataclass
class DecisionPoint:
    """One decision point of an ended episode (withdrawn ones included), as used by the update."""
    user_uuid: str
    psi: list      # baseline features
    terms: list    # centred treatment terms of the day's composition
    lags: list     # centred sums of R and of Q over the preceding days
    outcome: int   # r


@dataclass
class Posterior:
    """The Gaussian posterior of (theta, beta), in the order of NAMES."""
    names: list
    mean: list
    cov: list
    decision_points: int = 0
    participants: int = 0
    # Names of the baseline psi the version was fitted with; None for the versions written before
    # 11 November 2026, which used features.baseline_names(with_marks=True).
    baseline: list = None

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data):
        """The posterior stored in model_versions, or None for a version of the scheme of 14 October
        2026 (keys theta_mean, theta_cov), which the composition policy cannot use."""
        if "mean" not in data:
            return None
        return cls(**data)


def prior():
    """The posterior before any data: every coefficient N(0, tau^2)."""
    d = len(NAMES)
    return Posterior(names=list(NAMES), mean=[0.0] * d, cov=(np.eye(d) * design.PRIOR_SCALE_TAU ** 2).tolist())


def fit(points, baseline=None):
    """The posterior given the decision points of all finished episodes included in the update.

    The floating-point warnings of numpy are silenced because the BLAS of macOS raises them on
    finite inputs; the result is checked for finiteness instead."""
    if not points:
        return prior()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        posterior = solve(points)
    posterior.baseline = list(baseline) if baseline is not None else None
    if not (np.isfinite(posterior.mean).all() and np.isfinite(posterior.cov).all()):
        raise ValueError("the posterior is not finite")
    return posterior


def solve(points):
    """Fixed coefficients f = (alpha_0, theta, beta); the participant level alpha_i is eliminated by
    its Schur complement."""
    noise = design.NOISE_VARIANCE
    d = len(NAMES)
    p = len(points[0].psi)
    effects = slice(p, p + d)
    prior_precision = np.zeros(p + d)
    prior_precision[:p] = 1 / design.BASELINE_PRIOR_VARIANCE
    prior_precision[effects] = 1 / design.PRIOR_SCALE_TAU ** 2

    by_user = {}
    for point in points:
        by_user.setdefault(point.user_uuid, []).append(point)

    S = np.diag(prior_precision)   # precision of f after integrating out alpha_i
    h = np.zeros(p + d)            # the matching right-hand side
    for user_points in by_user.values():
        F = np.array([pt.psi + pt.terms + pt.lags for pt in user_points], dtype=float)
        r = np.array([pt.outcome for pt in user_points], dtype=float)
        # alpha_i ~ N(0, sigma_alpha^2): its precision given the participant's days, and the
        # cross terms with f.
        precision = len(user_points) / noise + 1 / design.INTERCEPT_VARIANCE
        cross = F.sum(axis=0) / noise
        S += F.T @ F / noise - np.outer(cross, cross) / precision
        h += F.T @ r / noise - cross * (r.sum() / noise) / precision

    cov_f = np.linalg.inv(S)
    cov_f = (cov_f + cov_f.T) / 2
    mean_f = cov_f @ h
    return Posterior(names=list(NAMES), mean=mean_f[effects].tolist(), cov=cov_f[effects, effects].tolist(),
                     decision_points=len(points), participants=len(by_user))
