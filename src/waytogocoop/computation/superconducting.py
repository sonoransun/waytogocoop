"""Superconducting gap modulation from moire patterns."""

from __future__ import annotations

import numpy as np


def gap_modulation(
    moire_pattern_2d: np.ndarray,
    delta_avg: float,
    delta_amplitude: float,
    phase_shift: float = np.pi,
) -> np.ndarray:
    """Compute a spatially varying superconducting gap field.

    The gap is modulated by the moire pattern intensity:

    .. math::

        \\Delta(\\mathbf{r}) = \\Delta_{\\text{avg}}
            + \\Delta_{\\text{amp}} \\cos(\\phi + \\pi \\, m(\\mathbf{r}))

    where *m(r)* is the normalized moire pattern (values in [0, 1]).

    Parameters
    ----------
    moire_pattern_2d : np.ndarray
        2D moire pattern, expected in [0, 1].
    delta_avg : float
        Average gap magnitude (meV).
    delta_amplitude : float
        Amplitude of gap modulation (meV).
    phase_shift : float
        Global phase offset (radians).  Default ``pi``.

    Returns
    -------
    np.ndarray
        2D array of gap values (meV), same shape as input.
    """
    return delta_avg + delta_amplitude * np.cos(
        phase_shift + np.pi * moire_pattern_2d
    )


def cpdm_amplitude(moire_period, coherence_length=20.0):
    """Estimate the Cooper-pair density modulation amplitude.

    A simple scaling ansatz:

    .. math::

        A_{\\text{CPDM}} = \\exp\\!\\left(-\\frac{\\xi}{L_m}\\right)

    where *xi* is the coherence length and *L_m* the moire period.
    When ``L_m >> xi`` the modulation saturates near 1; when ``L_m << xi``
    it is exponentially suppressed.

    Parameters
    ----------
    moire_period : float or array_like
        Moire superlattice period (Angstrom).
    coherence_length : float
        BCS coherence length (Angstrom).

    Returns
    -------
    float or np.ndarray
        Dimensionless CPDM amplitude in (0, 1].
    """
    moire_period = np.asarray(moire_period, dtype=float)
    result = np.where(
        np.isinf(moire_period) | (moire_period <= 0),
        0.0,
        np.exp(-coherence_length / np.where(moire_period > 0, moire_period, 1.0)),
    )
    return float(result) if result.ndim == 0 else result
