"""Reduced two-electron, one-nucleus Haberkorn reaction-yield kernel."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, pi, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve


@dataclass(frozen=True, slots=True)
class _ReducedIsotropicHaberkornSystem:
    """Parameters for the deliberately narrow reference spin system.

    The model contains two electron spins and one isotropically coupled spin-1/2
    nucleus on electron one. It starts in an electronic singlet with an
    unpolarized nucleus and includes electron Zeeman, scalar hyperfine, and
    Haberkorn singlet/triplet reaction terms. Nuclear Zeeman, exchange, dipolar
    coupling, relaxation, molecular motion, and orientation effects are absent.
    """

    hyperfine_hz: float
    singlet_rate_per_second: float
    triplet_rate_per_second: float
    electron_gyromagnetic_ratio_rad_per_second_tesla: float

    def __post_init__(self) -> None:
        values = (
            self.hyperfine_hz,
            self.singlet_rate_per_second,
            self.triplet_rate_per_second,
            self.electron_gyromagnetic_ratio_rad_per_second_tesla,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Haberkorn system parameters must be finite")
        if self.singlet_rate_per_second <= 0 or self.triplet_rate_per_second <= 0:
            raise ValueError("both Haberkorn reaction rates must be strictly positive")
        if self.electron_gyromagnetic_ratio_rad_per_second_tesla <= 0:
            raise ValueError("electron gyromagnetic-ratio magnitude must be strictly positive")


@dataclass(frozen=True, slots=True)
class _ReactionYields:
    """Infinite-time singlet and triplet reaction yields."""

    singlet: float
    triplet: float

    def __post_init__(self) -> None:
        if not isfinite(self.singlet) or not isfinite(self.triplet):
            raise ValueError("reaction yields must be finite")
        if not (0 <= self.singlet <= 1 and 0 <= self.triplet <= 1):
            raise ValueError("reaction yields must lie in [0, 1]")
        if not isclose(self.total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("reaction yields must conserve total probability")

    @property
    def total(self) -> float:
        """Return the conserved total reaction probability."""

        return self.singlet + self.triplet


def _solve_reaction_yields(
    system: _ReducedIsotropicHaberkornSystem,
    field_tesla: tuple[float, float, float],
) -> _ReactionYields:
    """Return infinite-time yields for one ideal Cartesian **B** vector in tesla."""

    if len(field_tesla) != 3 or not all(isfinite(component) for component in field_tesla):
        raise ValueError("magnetic flux-density B must contain three finite tesla components")

    identity = np.eye(2, dtype=np.complex128)
    sx = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128) / 2
    sy = np.array(((0.0, -1j), (1j, 0.0)), dtype=np.complex128) / 2
    sz = np.array(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128) / 2
    spin = (sx, sy, sz)

    electron_one = tuple(np.kron(np.kron(op, identity), identity) for op in spin)
    electron_two = tuple(np.kron(np.kron(identity, op), identity) for op in spin)
    nucleus = tuple(np.kron(np.kron(identity, identity), op) for op in spin)

    up = np.array((1.0, 0.0), dtype=np.complex128)
    down = np.array((0.0, 1.0), dtype=np.complex128)
    singlet = (np.kron(up, down) - np.kron(down, up)) / sqrt(2)
    singlet_electron_projector = np.outer(singlet, singlet.conj())
    singlet_projector = np.kron(singlet_electron_projector, identity)
    hilbert_identity = np.eye(8, dtype=np.complex128)
    triplet_projector = hilbert_identity - singlet_projector
    initial_density = np.kron(singlet_electron_projector, identity / 2)

    zeeman: NDArray[np.complex128] = np.zeros((8, 8), dtype=np.complex128)
    for component, first, second in zip(field_tesla, electron_one, electron_two, strict=True):
        zeeman += component * (first + second)
    hyperfine: NDArray[np.complex128] = np.zeros((8, 8), dtype=np.complex128)
    for first, nuclear_op in zip(electron_one, nucleus, strict=True):
        hyperfine += first @ nuclear_op
    hamiltonian = (
        system.electron_gyromagnetic_ratio_rad_per_second_tesla * zeeman
        + 2 * pi * system.hyperfine_hz * hyperfine
    )
    reaction_operator = (
        system.singlet_rate_per_second * singlet_projector
        + system.triplet_rate_per_second * triplet_projector
    )
    liouvillian = -1j * (
        np.kron(hilbert_identity, hamiltonian) - np.kron(hamiltonian.T, hilbert_identity)
    ) - 0.5 * (
        np.kron(hilbert_identity, reaction_operator)
        + np.kron(reaction_operator.T, hilbert_identity)
    )
    integrated_vector = solve(
        liouvillian,
        -initial_density.reshape(-1, order="F"),
        assume_a="gen",
        check_finite=True,
    )
    integrated_density: NDArray[np.complex128] = integrated_vector.reshape((8, 8), order="F")
    singlet_yield = float(
        (system.singlet_rate_per_second * np.trace(singlet_projector @ integrated_density)).real
    )
    triplet_yield = float(
        (system.triplet_rate_per_second * np.trace(triplet_projector @ integrated_density)).real
    )
    if not isclose(singlet_yield + triplet_yield, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ArithmeticError("Haberkorn reaction yields do not conserve total probability")
    if not (0 <= singlet_yield <= 1 and 0 <= triplet_yield <= 1):
        raise ArithmeticError("Haberkorn reaction yield lies outside [0, 1]")
    return _ReactionYields(singlet=singlet_yield, triplet=triplet_yield)
