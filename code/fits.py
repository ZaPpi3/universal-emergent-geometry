"""
Shared fitting utilities for finite-size scaling of lambda1(N).

Both candidate forms are fit as LINEAR least-squares problems (avoiding
any nonlinear-optimizer convergence ambiguity):

  power law:      y = a * N^-b        =>  ln(y) = ln(a) - b*ln(N)   (linear in ln a, b)
  log-corrected:  y = (a/N)(1+c/lnN)  =>  y = a*(1/N) + (ac)*(1/(N lnN))  (linear in a, ac)

R^2 is always reported in *real* (untransformed) y-space for both forms so
they are directly comparable, matching how the residual-pattern diagnostic
is used in the paper.

IMPORTANT CAVEAT (stated explicitly here and in the paper): lambda1(N) is
computed *exactly* via sparse diagonalization - there is no measurement
noise. Classical AIC/BIC assume i.i.d. Gaussian measurement noise, which
does not literally apply to noiseless exact data, so we do not report them
as formal significance tests. Instead, robustness of the extracted
exponents is assessed via a windowed (leave-smallest-N-out) refit: if an
exponent is a genuine asymptotic property rather than a small-N artifact,
it should stabilize as the smallest sizes are progressively dropped from
the fit.
"""
import numpy as np
from scipy.optimize import curve_fit


def fit_saturating(N, y):
    """y = yinf + a * N^-b  (nonlinear; allows a nonzero N->infinity asymptote).
    Included because TFIM lambda1(N) turns out to be far better described by
    a saturating asymptote than by either 2-parameter vanishing form - see
    the module-level note in run_all.py's stage_scaling_fits() section.
    """
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = [y.min(), 1.0, 1.0]
    popt, pcov = curve_fit(lambda n, yinf, a, b: yinf + a * n ** (-b), N, y,
                            p0=p0, maxfev=20000)
    yinf, a, b = popt
    perr = np.sqrt(np.diag(pcov))
    yhat = yinf + a * N ** (-b)
    resid = y - yhat
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(yinf=float(yinf), yinf_err=float(perr[0]), a=float(a), b=float(b),
                r2=float(r2), yhat=yhat, resid=resid)


def fit_power_law(N, y):
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.log(N)
    Y = np.log(y)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    slope, intercept = coef
    b = -slope
    a = np.exp(intercept)
    yhat = a * N ** (-b)
    resid = y - yhat
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(a=float(a), b=float(b), r2=float(r2), yhat=yhat, resid=resid)


def fit_log_corrected(N, y):
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    x1 = 1.0 / N
    x2 = 1.0 / (N * np.log(N))
    A = np.vstack([x1, x2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    p1, p2 = coef  # p1 = a, p2 = a*c
    a = p1
    c = p2 / p1 if abs(p1) > 1e-14 else float("nan")
    yhat = A @ coef
    resid = y - yhat
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(a=float(a), c=float(c), r2=float(r2), yhat=yhat, resid=resid)


def windowed_fit_stability(N, y, fit_func, min_points=4):
    """Progressively drop the smallest system sizes and refit; returns a
    list of (N_min, fitted_param, r2) so the stability of the extracted
    exponent/coefficient can be assessed as a function of fitting window.
    """
    N = np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(N)
    N, y = N[order], y[order]
    results = []
    for start in range(len(N) - min_points + 1):
        Nw, yw = N[start:], y[start:]
        if len(Nw) < min_points:
            break
        fit = fit_func(Nw, yw)
        key = "b" if "b" in fit else "c"
        results.append((float(N[start]), fit[key], fit["r2"]))
    return results
