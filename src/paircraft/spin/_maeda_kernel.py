"""Provenance-neutral numerical kernel for the reduced five-nucleus Maeda model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

_HILBERT_DIMS: Final = (2, 2, 3, 2, 2, 2, 2)
_HILBERT_DIM: Final = 192
_NUCLEAR_DIM: Final = 48
LIOUVILLE_DIM: Final = _HILBERT_DIM**2


@dataclass(frozen=True, slots=True)
class KernelHyperfine:
    """One signed isotropic equivalent-field coupling used by the kernel."""

    radical_index: int
    spin_quantum_number: float
    equivalent_field_tesla: float

    def __post_init__(self) -> None:
        if self.radical_index not in {0, 1}:
            raise ValueError("kernel hyperfine radical index must be zero or one")
        if self.spin_quantum_number not in {0.5, 1.0}:
            raise ValueError("kernel supports only spin-1/2 and spin-1 nuclei")
        if not isfinite(self.equivalent_field_tesla):
            raise ValueError("kernel hyperfine field must be finite")


@dataclass(frozen=True, slots=True)
class MaedaKernelInputs:
    """Complete numerical inputs without scientific provenance or claim semantics."""

    hyperfine_parameters: tuple[KernelHyperfine, ...]
    electron_gyromagnetic_ratio_rad_per_second_tesla: float
    forward_rate_per_second: float
    back_recombination_rate_per_second: float
    st_dephasing_rate_per_second: float

    def __post_init__(self) -> None:
        if len(self.hyperfine_parameters) != 5:
            raise ValueError("kernel requires exactly five hyperfine nuclei")
        gamma = self.electron_gyromagnetic_ratio_rad_per_second_tesla
        if not isfinite(gamma) or gamma <= 0:
            raise ValueError("kernel electron gyromagnetic ratio must be finite and positive")
        rates = (
            self.forward_rate_per_second,
            self.back_recombination_rate_per_second,
            self.st_dephasing_rate_per_second,
        )
        if any(not isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("kernel rates must be finite and nonnegative")
        if self.forward_rate_per_second <= 0:
            raise ValueError("kernel forward rate must be strictly positive")


@dataclass(frozen=True, slots=True)
class MaedaKernelResult:
    """Numerical outputs without a model identity, provenance, or scientific claim."""

    secondary_pair: float
    singlet_back_recombination: float
    relative_residual: float
    iterations: int


def _spin_operators(spin: float) -> tuple[NDArray[np.complex128], ...]:
    dimension = round(2 * spin + 1)
    magnetic_quantum_numbers = np.arange(spin, -spin - 1, -1, dtype=float)
    z = np.diag(magnetic_quantum_numbers).astype(np.complex128)
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(1, dimension):
        m_value = magnetic_quantum_numbers[column]
        raising[column - 1, column] = np.sqrt(spin * (spin + 1) - m_value * (m_value + 1))
    lowering = raising.T.conj()
    return (raising + lowering) / 2, (raising - lowering) / (2j), z


def _embed(operator: NDArray[np.complex128], position: int) -> sparse.csr_matrix:
    result = sparse.csr_matrix([[1.0 + 0j]])
    for index, dimension in enumerate(_HILBERT_DIMS):
        factor = (
            sparse.csr_matrix(operator)
            if index == position
            else sparse.eye(dimension, format="csr", dtype=np.complex128)
        )
        result = sparse.kron(result, factor, format="csr")
    return result


def _build_operators() -> tuple[
    tuple[tuple[sparse.csr_matrix, ...], tuple[sparse.csr_matrix, ...]],
    tuple[tuple[sparse.csr_matrix, ...], ...],
    sparse.csr_matrix,
]:
    half = _spin_operators(0.5)
    one = _spin_operators(1.0)
    electrons = (
        tuple(_embed(operator, 0) for operator in half),
        tuple(_embed(operator, 1) for operator in half),
    )
    nuclei = (
        tuple(_embed(operator, 2) for operator in one),
        tuple(_embed(operator, 3) for operator in half),
        tuple(_embed(operator, 4) for operator in half),
        tuple(_embed(operator, 5) for operator in half),
        tuple(_embed(operator, 6) for operator in half),
    )
    up = np.array((1.0, 0.0), dtype=np.complex128)
    down = np.array((0.0, 1.0), dtype=np.complex128)
    singlet = (np.kron(up, down) - np.kron(down, up)) / np.sqrt(2)
    electron_projector = sparse.csr_matrix(np.outer(singlet, singlet.conj()))
    singlet_projector = sparse.kron(
        electron_projector,
        sparse.eye(_NUCLEAR_DIM),
        format="csr",
    )
    return electrons, nuclei, singlet_projector


_ELECTRONS, _NUCLEI, _SINGLET_PROJECTOR = _build_operators()


def _hamiltonian(
    inputs: MaedaKernelInputs,
    field_tesla: tuple[float, float, float],
) -> sparse.csr_matrix:
    gamma = inputs.electron_gyromagnetic_ratio_rad_per_second_tesla
    hamiltonian = sparse.csr_matrix((_HILBERT_DIM, _HILBERT_DIM), dtype=np.complex128)
    for component, first, second in zip(field_tesla, *_ELECTRONS, strict=True):
        hamiltonian = hamiltonian + gamma * component * (first + second)
    for parameter, nuclear_operators in zip(
        inputs.hyperfine_parameters,
        _NUCLEI,
        strict=True,
    ):
        electron_operators = _ELECTRONS[parameter.radical_index]
        for electron, nucleus in zip(electron_operators, nuclear_operators, strict=True):
            hamiltonian = hamiltonian + gamma * parameter.equivalent_field_tesla * (
                electron @ nucleus
            )
    return hamiltonian.tocsr()


def st_dephasing(
    singlet: NDArray[np.complex128],
    triplet: NDArray[np.complex128],
    density: NDArray[np.complex128],
    rate_per_second: float,
) -> NDArray[np.complex128]:
    """Damp only the singlet-triplet and triplet-singlet coherence blocks."""

    return -rate_per_second * (singlet @ density @ triplet + triplet @ density @ singlet)


def solve_maeda_kernel(
    inputs: MaedaKernelInputs,
    field_tesla: tuple[float, float, float],
    *,
    relative_tolerance: float = 1e-8,
    restart: int = 100,
    maximum_restart_cycles: int = 50,
) -> MaedaKernelResult:
    """Solve one neutral numerical input set at an ideal magnetic flux density **B**."""

    if len(field_tesla) != 3 or not all(isfinite(value) for value in field_tesla):
        raise ValueError("magnetic flux-density B must contain three finite tesla components")
    if not isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative tolerance must be finite and lie in (0, 1)")
    if restart <= 0 or maximum_restart_cycles <= 0:
        raise ValueError("GMRES iteration limits must be strictly positive")

    hamiltonian = _hamiltonian(inputs, field_tesla)
    energies, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
    singlet = eigenvectors.conj().T @ _SINGLET_PROJECTOR.toarray() @ eigenvectors
    triplet = np.eye(_HILBERT_DIM, dtype=np.complex128) - singlet
    initial_density = singlet / _NUCLEAR_DIM
    forward = inputs.forward_rate_per_second
    back = inputs.back_recombination_rate_per_second
    dephasing = inputs.st_dephasing_rate_per_second

    def apply(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
        density = vector.reshape((_HILBERT_DIM, _HILBERT_DIM), order="F")
        commutator = (energies[:, None] - energies[None, :]) * density
        singlet_density = singlet @ density
        density_singlet = density @ singlet
        derivative = (
            -1j * commutator
            - forward * density
            - 0.5 * back * (singlet_density + density_singlet)
            + st_dephasing(singlet, triplet, density, dephasing)
        )
        return np.asarray(derivative).reshape(-1, order="F")

    operator = LinearOperator(
        (LIOUVILLE_DIM, LIOUVILLE_DIM),
        matvec=apply,
        dtype=np.complex128,
    )
    singlet_diagonal = np.diag(singlet).real
    triplet_diagonal = np.diag(triplet).real
    diagonal = (
        -1j * (energies[:, None] - energies[None, :])
        - forward
        - 0.5 * back * (singlet_diagonal[:, None] + singlet_diagonal[None, :])
        - dephasing
        * (
            singlet_diagonal[:, None] * triplet_diagonal[None, :]
            + triplet_diagonal[:, None] * singlet_diagonal[None, :]
        )
    ).reshape(-1, order="F")
    preconditioner = LinearOperator(
        (LIOUVILLE_DIM, LIOUVILLE_DIM),
        matvec=lambda vector: vector / diagonal,
        dtype=np.complex128,
    )
    residual_history: list[float] = []
    right_hand_side: NDArray[np.complex128] = -initial_density.reshape(-1, order="F")
    gmres_complex = cast(
        Callable[..., tuple[NDArray[np.complex128], int]],
        gmres,
    )
    integrated_vector, info = gmres_complex(
        operator,
        right_hand_side,
        M=preconditioner,
        rtol=relative_tolerance,
        atol=0.0,
        restart=restart,
        maxiter=maximum_restart_cycles,
        callback=lambda residual: residual_history.append(float(residual)),
        callback_type="pr_norm",
    )
    if info != 0 or not residual_history:
        raise ArithmeticError(f"Maeda GMRES failed to converge; scipy info={info}")
    true_residual = float(
        np.linalg.norm(apply(integrated_vector) - right_hand_side) / np.linalg.norm(right_hand_side)
    )
    if not isfinite(true_residual) or true_residual > relative_tolerance:
        raise ArithmeticError("Maeda GMRES true residual exceeds the requested relative tolerance")
    integrated = integrated_vector.reshape((_HILBERT_DIM, _HILBERT_DIM), order="F")
    return MaedaKernelResult(
        secondary_pair=float((forward * np.trace(integrated)).real),
        singlet_back_recombination=float((back * np.trace(singlet @ integrated)).real),
        relative_residual=true_residual,
        iterations=len(residual_history),
    )
