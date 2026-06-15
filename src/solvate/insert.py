#!/usr/bin/env python
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build universes from template molecules."""

import logging
from typing import Optional, Callable

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

from .models import empty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

epsilon_0 = 8.8541878128e-22  # F/Å
kB = 1.380649e-23  # J/K
e = 1.602176634e-19  # C


def _renumber_projectile_resids(
    SolvatedUniverse: mda.Universe, nAtomsTarget: int
) -> mda.Universe:
    """Renumber residues after the target so resids are contiguous and monotonic.

    Target residues (the first `nAtomsTarget` atoms) keep their original resids.
    Residues from the projectile atoms are renumbered starting at
    ``target.resids[-1] + 1`` (or ``1`` when the target is empty), so the
    returned universe has no gaps or duplicates among the projectile resids.
    """
    n_total_res = len(SolvatedUniverse.residues)
    if nAtomsTarget == 0:
        start = 1
        n_target_res = 0
    else:
        target = SolvatedUniverse.atoms[:nAtomsTarget]
        # Use max() rather than [-1] so we don't collide with an
        # out-of-order target resid (e.g. user-supplied [5, 2, 3]).
        start = target.residues.resids.max() + 1
        n_target_res = len(target.residues)
    n_proj_res = n_total_res - n_target_res
    if n_proj_res > 0:
        projectile = SolvatedUniverse.atoms[nAtomsTarget:]
        projectile.residues.resids = np.arange(start, start + n_proj_res)
    return SolvatedUniverse


def tile_universe(
    universe: mda.Universe,
    n_x: int,
    n_y: int,
    n_z: int,
) -> mda.Universe:
    """Returns a new Universe with `n_x * n_y * n_z` copies of the input."""
    box = universe.dimensions[:3]
    copied = []
    i = 0
    for x in tqdm(range(n_x)):
        for y in range(n_y):
            for z in range(n_z):
                u_ = universe.copy()
                move_by = box * (x, y, z)
                u_.residues.resids += len(universe.residues) * i
                u_.atoms.translate(move_by)
                copied.append(u_.atoms)
                i += 1

    new_universe = mda.Merge(*copied)
    new_box = box * (n_x, n_y, n_z)
    new_universe.dimensions = list(new_box) + [90] * 3
    return new_universe


def pos_random(InsertionDomain: np.ndarray) -> np.ndarray:
    """Returns a random position within the given domain."""
    return np.array(
        np.random.rand(3) * (InsertionDomain[3:6] - InsertionDomain[0:3])
        + InsertionDomain[0:3],
        dtype=np.float32,
    )


def rot_random() -> tuple:
    """Returns a random rotation angle and vector, sampled uniformly on a sphere."""
    u_1, u_2, u_3 = np.random.rand(3)

    theta, phi = np.arccos(2 * u_1 - 1), 2 * np.pi * u_2

    rot_vec = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    rot_angle = 360 * u_3

    return rot_angle, rot_vec


def SolvateCylinder(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    density: Optional[float] = None,
    pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    min: float = 0,
    max: Optional[float] = None,
    dim: int = 2,
    distance: float = 1.25,
    n_tries: int = 1000,
    fudge_factor: float = 1,
) -> mda.Universe:
    """Fill a cylindrical region of a target with copies of a projectile.

    Internally builds a small saturated patch of projectiles, tiles it across
    the cylinder's bounding box, then prunes any copies that fall outside the
    cylinder or overlap atoms in ``TargetUniverse``. For most use cases this is
    orders of magnitude faster than :func:`InsertCylinder`.

    Parameters
    ----------
    TargetUniverse : MDAnalysis.core.universe.Universe
        Universe to solvate. May be empty; in that case
        ``TargetUniverse.dimensions`` still defines the simulation cell of the
        returned universe.
    ProjectileUniverse : MDAnalysis.core.universe.Universe
        Molecule that is inserted repeatedly.
    n : int, default 1
        Number of projectile copies to insert. Ignored when ``density`` is
        given.
    density : float, optional
        Target number density of projectiles, in molecules / Å³. When set,
        ``n`` is computed from the cylinder volume and ``n`` is ignored.
    pos : array_like of shape (3,), optional
        Centre of the cylinder, in Å. Defaults to the centre of geometry of
        ``TargetUniverse`` (or the centre of its box if the target is empty).
        The component along ``dim`` is overridden by ``min``.
    radius : float, optional
        Cylinder radius, in Å. Defaults to half of the smallest box edge of
        ``TargetUniverse``.
    min, max : float, optional
        Lower and upper bound of the cylinder along axis ``dim``, in Å.
        ``max`` defaults to ``TargetUniverse.dimensions[dim]``.
    dim : {0, 1, 2}, default 2
        Index of the axis along which the cylinder extends (0 = x, 1 = y,
        2 = z).
    distance : float, default 1.25
        Minimum allowed distance (Å) between an inserted projectile and any
        atom of the target.
    n_tries : int, default 1000
        Maximum number of random placement attempts used when building the
        seed patch.
    fudge_factor : float, default 1.0
        Multiplier on the number of projectiles packed into the seed patch.
        Increased automatically and the function recurses when too few
        projectiles survive overlap pruning.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        New universe containing the original target atoms followed by the
        inserted projectile copies.

    See Also
    --------
    InsertCylinder : Slower per-molecule variant with full placement control.
    SolvatePlanar : Equivalent for rectangular regions.
    """
    logger.info(f"The fudge factor is {fudge_factor}")
    if max is None:
        max = TargetUniverse.dimensions[dim]

    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos is None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget[:3] / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()
    pos[dim] = min

    if radius is None:
        radius = np.min(dimensionsTarget) / 2

    if density is not None:
        n = np.floor(density * (2 * radius) ** 2 * (max - min))
        solvate_by_density_flag = True
    else:
        solvate_by_density_flag = False

    dimensionsTarget = TargetUniverse.dimensions.copy()

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

    InsertionDomain = np.array(
        [pos[0] - radius, pos[1] - radius, min, pos[0] + radius, pos[1] + radius, max],
        dtype=np.float32,
    )

    InsertionVolume = (max - min) * np.pi * radius**2
    density = n / InsertionVolume

    SolvatedUniverse = SolvatePlanar(
        TargetUniverse,
        ProjectileUniverse,
        0,
        density,
        xmin=InsertionDomain[0],
        ymin=InsertionDomain[1],
        zmin=InsertionDomain[2],
        xmax=InsertionDomain[3],
        ymax=InsertionDomain[4],
        zmax=InsertionDomain[5],
        distance=distance,
        n_tries=n_tries,
        fudge_factor=fudge_factor,
    )
    dims = SolvatedUniverse.dimensions
    TargetAtoms = SolvatedUniverse.atoms[:nAtomsTarget]
    ProjectileAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
    atomsInside = (
        np.linalg.norm((ProjectileAtoms.positions - pos)[:, :2], axis=1) < radius
    )
    if TargetAtoms.n_atoms == 0:
        SolvatedUniverse = ProjectileAtoms[atomsInside].residues.atoms
    else:
        SolvatedUniverse = mda.Merge(
            TargetAtoms, ProjectileAtoms[atomsInside].residues.atoms
        )
    SolvatedUniverse.dimensions = dims
    logger.info("Resulting number of atoms:", SolvatedUniverse.atoms.n_atoms)
    logger.info(
        "Resulting number of projectiles:",
        (SolvatedUniverse.atoms.n_atoms - nAtomsTarget) / nAtomsProjectile,
    )

    missingProjectiles = int(
        ((n * nAtomsProjectile + nAtomsTarget) - SolvatedUniverse.atoms.n_atoms)
        / nAtomsProjectile
    )
    logger.info("Missing", missingProjectiles, "Projectiles.")
    if solvate_by_density_flag:
        logger.info(
            f" {SolvatedUniverse.atoms.n_atoms - nAtomsTarget} projectiles inserted"
        )
        return _renumber_projectile_resids(SolvatedUniverse, nAtomsTarget)
    if missingProjectiles > 0:
        logger.info("Missing", missingProjectiles, "Projectiles.")
        logger.info("Adjusting fudge factor and trying again.")
        new_fudge_factor = fudge_factor + 0.5
        return SolvateCylinder(
            TargetUniverse,
            ProjectileUniverse,
            n,
            density=None,
            pos=pos,
            radius=radius,
            min=min,
            max=max,
            dim=dim,
            distance=distance,
            n_tries=n_tries,
            fudge_factor=new_fudge_factor,
        )

    if missingProjectiles < 0:
        nonTargetAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
        logger.info("Too many projectiles inserted:", -missingProjectiles)
        logger.info(nonTargetAtoms.n_atoms)
        logger.info(nonTargetAtoms.residues.n_residues)
        logger.info(np.unique(nonTargetAtoms.residues.resids).shape)
        logger.info("Removing", -missingProjectiles, "randomly selected projectiles.")
        ToBeRemoved = nonTargetAtoms.residues[
            np.random.choice(
                np.arange(len(nonTargetAtoms.residues)),
                -missingProjectiles,
                replace=False,
            )
        ]
        SolvatedUniverse = mda.Merge(SolvatedUniverse.atoms - ToBeRemoved.atoms)
        SolvatedUniverse.dimensions = dimensionsTarget
        logger.info("Final number of atoms:", SolvatedUniverse.atoms.n_atoms)
        return _renumber_projectile_resids(SolvatedUniverse, nAtomsTarget)
    logger.info("All projectiles inserted correctly")
    return _renumber_projectile_resids(SolvatedUniverse, nAtomsTarget)


def SolvatePlanar(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    density: Optional[float] = None,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    zmax: Optional[float] = None,
    distance: float = 1.25,
    solvate_factor: int = 100,
    fudge_factor: float = 1.0,
    n_tries: int = 1000,
) -> mda.Universe:
    """Fill a rectangular region of a target with copies of a projectile.

    Internally builds a small saturated patch of projectiles, tiles it across
    the insertion box, then prunes any copies that overlap atoms in
    ``TargetUniverse``. This is orders of magnitude faster than
    :func:`InsertPlanar` for large solvent counts and is the recommended way
    to solvate a target with thousands of solvent molecules.

    Parameters
    ----------
    TargetUniverse : MDAnalysis.core.universe.Universe
        Universe to solvate. May be empty; ``TargetUniverse.dimensions``
        defines the simulation cell of the returned universe.
    ProjectileUniverse : MDAnalysis.core.universe.Universe
        Molecule that is inserted repeatedly.
    n : int, default 1
        Number of projectile copies to insert. Ignored when ``density`` is
        given.
    density : float, optional
        Target number density of projectiles, in molecules / Å³. When set,
        ``n`` is computed from the volume of the insertion box and ``n`` is
        ignored.
    xmin, ymin, zmin : float, default 0
        Lower bounds of the insertion box, in Å.
    xmax, ymax, zmax : float, optional
        Upper bounds of the insertion box, in Å. Each defaults to the
        corresponding component of ``TargetUniverse.dimensions``.
    distance : float, default 1.25
        Minimum allowed distance (Å) between an inserted projectile and any
        atom of the target. Tile copies closer than ``distance`` are removed
        after tiling.
    solvate_factor : int, default 100
        Target number of projectiles in each tiled sub-box. Larger values
        reduce the number of tiles and the cost of the per-tile saturation
        step; smaller values reduce peak memory.
    fudge_factor : float, default 1.0
        Multiplier on ``solvate_factor`` controlling how aggressively the
        seed patch is packed. Increased automatically and the function
        recurses when too few projectiles survive overlap pruning.
    n_tries : int, default 1000
        Base number of random placement attempts used when packing the seed
        patch (internally scaled by 1000).

    Returns
    -------
    MDAnalysis.core.universe.Universe
        New universe containing the original target atoms followed by the
        inserted projectile copies.

    See Also
    --------
    InsertPlanar : Slower per-molecule variant with full placement control.
    SolvateCylinder : Equivalent for cylindrical regions.
    """
    # Use no fewer than 20 atoms for solvation
    SOLVATION_THRESHOLD = 20

    if xmax is None:
        xmax = TargetUniverse.dimensions[0]
    if ymax is None:
        ymax = TargetUniverse.dimensions[1]
    if zmax is None:
        zmax = TargetUniverse.dimensions[2]
    if xmin is None:
        xmin = 0
    if ymin is None:
        ymin = 0
    if zmin is None:
        zmin = 0

    InsertionDomain = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomainSize = InsertionDomain[3:6] - InsertionDomain[0:3]
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if density is not None:
        n = np.floor(
            density
            * InsertionDomainSize[0]
            * InsertionDomainSize[1]
            * InsertionDomainSize[2]
        )

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

    logger.info(f"Should solvate {n} Projectiles")
    x = np.ceil((n / (solvate_factor * fudge_factor)) ** (1 / 3)).astype(int)

    if x <= 1:
        x = 1
        logger.info(f"Solvation factor: {solvate_factor}")
        logger.info(f"Best tiling is {x}x{x}x{x}.")

        return _renumber_projectile_resids(
            InsertPlanar(
                TargetUniverse,
                ProjectileUniverse,
                n,
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax,
                distance,
                n_tries,
            ),
            nAtomsTarget,
        )
    if n / (x**3) < SOLVATION_THRESHOLD and x > 2:
        x -= 1

    real_solvate_factor = n / (x**3)

    logger.info(f"Solvation factor: {solvate_factor}")
    logger.info(f"Best tiling is {x}x{x}x{x}.")

    real_solvate_factor = np.ceil(real_solvate_factor * fudge_factor).astype(int)

    logger.info("Real solvation factor is", real_solvate_factor)
    logger.info(
        "This results in a total of",
        x**3 * (real_solvate_factor),
        "projectiles in the solvate box",
    )
    solvate_box_dimensions = np.concatenate(
        [InsertionDomainSize / x, dimensionsTarget[3:6]]
    )

    solvate_box = InsertPlanar(
        empty(solvate_box_dimensions),
        ProjectileUniverse,
        real_solvate_factor,
        distance=distance,
        n_tries=n_tries * 1000,
    )

    # We tile the small box to make a big box that is big enough to contain
    # the insertion domain
    logger.info("Tiling solvate box...")
    big_solvate_box = tile_universe(solvate_box, x, x, x)

    # Shift the solvate box to the beginning of the insertion domain
    big_solvate_box.atoms.translate(InsertionDomain[0:3])

    logger.info("Inserting solvate box into target universe...")

    nAtomsSolvate = big_solvate_box.atoms.n_atoms

    logger.info("Target atoms:", nAtomsTarget)
    logger.info("Projectile atoms:", nAtomsSolvate)

    if nAtomsTarget == 0:
        SolvatedUniverse = big_solvate_box
    else:
        SolvatedUniverse = mda.Merge(TargetUniverse.atoms, big_solvate_box.atoms)
    SolvatedUniverse.dimensions = dimensionsTarget
    target = SolvatedUniverse.atoms[0:nAtomsTarget]
    projectile = SolvatedUniverse.atoms[-nAtomsSolvate:]

    logger.info("Search for overlapping atoms...")

    ns = mda.lib.NeighborSearch.AtomNeighborSearch(
        projectile, SolvatedUniverse.dimensions
    )
    touching_atoms = ns.search(target, distance, level="R").atoms
    if touching_atoms.n_atoms > 0:
        # touching_atoms = touching_atoms.intersection(projectile).residues.atoms
        # if touching_atoms.n_atoms / nAtomsProjectile:

        logger.info(
            "Removing touching projectiles:", touching_atoms.n_atoms / nAtomsProjectile
        )
        SolvatedUniverse = mda.Merge(SolvatedUniverse.atoms - touching_atoms)
        SolvatedUniverse.dimensions = dimensionsTarget
    logger.info("Resulting number of atoms:", SolvatedUniverse.atoms.n_atoms)
    logger.info("Expected number of atoms:", n * nAtomsProjectile + nAtomsTarget)
    missingProjectiles = int(
        ((n * nAtomsProjectile + nAtomsTarget) - SolvatedUniverse.atoms.n_atoms)
        / nAtomsProjectile
    )

    if density is not None:
        logger.info(
            f" {SolvatedUniverse.atoms.n_atoms - nAtomsTarget} projectiles inserted"
        )
        return _renumber_projectile_resids(SolvatedUniverse, nAtomsTarget)
    if missingProjectiles > 0:
        logger.info("Missing", missingProjectiles, "Projectiles.")
        logger.info("Adjusting fudge factor and trying again.")
        return SolvatePlanar(
            TargetUniverse,
            ProjectileUniverse,
            n,
            density,
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            distance,
            solvate_factor,
            fudge_factor + 10 * missingProjectiles / n,
            n_tries,
        )
    if missingProjectiles < 0:
        nonTargetAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
        logger.info("Too many projectiles inserted:", -missingProjectiles)
        logger.info(nonTargetAtoms.n_atoms)
        logger.info(nonTargetAtoms.residues.n_residues)
        logger.info(np.unique(nonTargetAtoms.residues.resids).shape)
        logger.info("Removing", -missingProjectiles, "randomly selected projectiles.")
        ToBeRemoved = nonTargetAtoms.residues[
            np.random.choice(
                np.arange(len(nonTargetAtoms.residues)),
                -missingProjectiles,
                replace=False,
            )
        ]
        SolvatedUniverse = mda.Merge(SolvatedUniverse.atoms - ToBeRemoved.atoms)
        SolvatedUniverse.dimensions = dimensionsTarget
        logger.info("Final number of atoms:", SolvatedUniverse.atoms.n_atoms)
        return _renumber_projectile_resids(SolvatedUniverse, nAtomsTarget)
    logger.info("All projectiles inserted correctly")
    return _renumber_projectile_resids(SolvatedUniverse, nAtomsTarget)


def InsertPlanar(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    zmax: Optional[float] = None,
    distance: float = 1.25,
    n_tries: int = 1000,
) -> mda.Universe:
    """Insert ``n`` copies of a projectile into a rectangular region.

    Each projectile is placed at a random position and orientation inside the
    axis-aligned box defined by ``(xmin, ymin, zmin)`` and
    ``(xmax, ymax, zmax)``. Up to ``n_tries`` placement attempts are made per
    projectile; a :class:`RuntimeError` is raised if no overlap-free position
    is found.

    Parameters
    ----------
    TargetUniverse : MDAnalysis.core.universe.Universe
        Universe to insert into. May be empty; ``TargetUniverse.dimensions``
        then defines the simulation cell of the returned universe.
    ProjectileUniverse : MDAnalysis.core.universe.Universe
        Molecule that is inserted repeatedly.
    n : int, default 1
        Number of projectile copies to insert.
    xmin, ymin, zmin : float, default 0
        Lower bounds of the insertion box, in Å.
    xmax, ymax, zmax : float, optional
        Upper bounds of the insertion box, in Å. Each defaults to the
        corresponding component of ``TargetUniverse.dimensions``.
    distance : float, default 1.25
        Minimum allowed distance (Å) between the inserted projectile and any
        existing atom in the target.
    n_tries : int, default 1000
        Maximum number of random placement attempts per projectile.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        New universe containing the target atoms followed by the inserted
        projectile copies.

    Raises
    ------
    RuntimeError
        If no overlap-free position is found within ``n_tries`` attempts for a
        given projectile.

    See Also
    --------
    SolvatePlanar : Fast variant for many projectiles.
    InsertCylinder, InsertSphere
    """
    nAtomsTargetOriginal = TargetUniverse.atoms.n_atoms
    InsertionDomain = [xmin, ymin, zmin, xmax, ymax, zmax]
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomain = np.array(InsertionDomain)
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    if TargetUniverse.atoms.n_atoms == 0:
        TargetUniverse = ProjectileUniverse.copy()
        TargetUniverse.dimensions = dimensionsTarget
        TargetUniverse.atoms.translate(
            pos_random(InsertionDomain) - ProjectileUniverse.atoms.center_of_geometry()
        )
        TargetUniverse.atoms.rotateby(*rot_random())
        n -= 1

    for _N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms

        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]
        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target, dimensionsTarget)

        for _attempt in range(n_tries):
            projectile.translate(
                pos_random(InsertionDomain) - projectile.atoms.center_of_geometry()
            )

            projectile.rotateby(*rot_random())

            if len(ns.search(projectile, distance)) == 0:
                break
        else:
            raise RuntimeError(
                "Error: No suitable position found,\
                maybe you are trying to insert to many particles? Aborting."
            )

    return _renumber_projectile_resids(TargetUniverse, nAtomsTargetOriginal)


def InsertCylinder(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    min: float = 0,
    max: Optional[float] = None,
    dim: int = 2,
    distance: float = 1.25,
    n_tries: int = 1000,
) -> mda.Universe:
    """Insert ``n`` copies of a projectile into a cylindrical region.

    Each projectile is placed at a random position and orientation inside the
    cylinder centred at ``pos`` with radius ``radius``, extending from ``min``
    to ``max`` along axis ``dim``. Up to ``n_tries`` placement attempts are
    made per projectile; a :class:`RuntimeError` is raised if no overlap-free
    position is found.

    Parameters
    ----------
    TargetUniverse : MDAnalysis.core.universe.Universe
        Universe to insert into. May be empty.
    ProjectileUniverse : MDAnalysis.core.universe.Universe
        Molecule that is inserted repeatedly.
    n : int, default 1
        Number of projectile copies to insert.
    pos : array_like of shape (3,), optional
        Centre of the cylinder, in Å. Defaults to the centre of geometry of
        ``TargetUniverse`` (or the centre of its box if the target is empty).
        The component along ``dim`` is overridden by ``min``.
    radius : float, optional
        Cylinder radius, in Å. Defaults to half of the smallest box edge of
        ``TargetUniverse``.
    min, max : float, optional
        Lower and upper bound of the cylinder along axis ``dim``, in Å.
        ``max`` defaults to ``TargetUniverse.dimensions[dim]``.
    dim : {0, 1, 2}, default 2
        Index of the axis along which the cylinder extends (0 = x, 1 = y,
        2 = z).
    distance : float, default 1.25
        Minimum allowed distance (Å) between the inserted projectile and any
        existing atom in the target.
    n_tries : int, default 1000
        Maximum number of random placement attempts per projectile.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        New universe containing the target atoms followed by the inserted
        projectile copies.

    Raises
    ------
    RuntimeError
        If no overlap-free position is found within ``n_tries`` attempts for a
        given projectile.

    See Also
    --------
    SolvateCylinder : Fast variant for many projectiles.
    InsertPlanar, InsertSphere
    """
    nAtomsTargetOriginal = TargetUniverse.atoms.n_atoms
    if max is None:
        max = TargetUniverse.dimensions[dim]

    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos is None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()
    pos[dim] = min

    if radius is None:
        radius = np.min(dimensionsTarget) / 2

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    for _N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        # Generate coordinates and check for overlap
        for _attempt in range(n_tries):
            projectile.rotateby(*rot_random())

            r = radius * np.sqrt(np.random.rand())
            phi, z = np.random.rand(2) * [2 * np.pi, (max - min)]
            newcoord = np.roll([r * np.cos(phi), r * np.sin(phi), z], dim - 2) + pos

            projectile.translate(newcoord - projectile.atoms.center_of_geometry())

            if len(ns.search(projectile, distance)) == 0:
                break
        else:
            raise RuntimeError(
                "Error: No suitable position found,\
                maybe you are trying to insert too many particles? Aborting."
            )

    return _renumber_projectile_resids(TargetUniverse, nAtomsTargetOriginal)


def InsertSphere(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    distance: float = 1.25,
    n_tries: int = 1000,
) -> mda.Universe:
    """Insert ``n`` copies of a projectile into a spherical region.

    Each projectile is placed at a uniformly random position inside the
    sphere centred at ``pos`` with radius ``radius`` and a random
    orientation. Up to ``n_tries`` placement attempts are made per projectile;
    a :class:`RuntimeError` is raised if no overlap-free position is found.

    Parameters
    ----------
    TargetUniverse : MDAnalysis.core.universe.Universe
        Universe to insert into. May be empty.
    ProjectileUniverse : MDAnalysis.core.universe.Universe
        Molecule that is inserted repeatedly.
    n : int, default 1
        Number of projectile copies to insert.
    pos : array_like of shape (3,), optional
        Centre of the sphere, in Å. Defaults to the centre of geometry of
        ``TargetUniverse`` (or the centre of its box if the target is empty).
    radius : float, optional
        Sphere radius, in Å. Defaults to half of the smallest box edge of
        ``TargetUniverse``.
    distance : float, default 1.25
        Minimum allowed distance (Å) between the inserted projectile and any
        existing atom in the target.
    n_tries : int, default 1000
        Maximum number of random placement attempts per projectile.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        New universe containing the target atoms followed by the inserted
        projectile copies.

    Raises
    ------
    RuntimeError
        If no overlap-free position is found within ``n_tries`` attempts for a
        given projectile.

    See Also
    --------
    InsertPlanar, InsertCylinder
    """

    def rand_spherical(radius: float = 1.0) -> np.ndarray:
        u = np.random.rand()
        v = np.random.rand()

        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        r = radius * np.power(np.random.rand(), 1 / 3)

        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        return np.array([x, y, z])

    nAtomsTargetOriginal = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos is None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget[:3] / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()

    if radius is None:
        radius = np.min(dimensionsTarget) / 2

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    if TargetUniverse.atoms.n_atoms == 0:
        TargetUniverse = ProjectileUniverse.copy()
        TargetUniverse.dimensions = dimensionsTarget
        TargetUniverse.atoms.translate(
            pos + rand_spherical(radius) - TargetUniverse.atoms.center_of_geometry()
        )
        TargetUniverse.atoms.rotateby(*rot_random())
        n -= 1

    for _N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        # Generate coordinates and check for overlap
        for _attempt in range(n_tries):
            projectile.rotateby(*rot_random())
            newcoord = rand_spherical(radius) + pos
            projectile.translate(newcoord - projectile.atoms.center_of_geometry())
            if len(ns.search(projectile, distance)) == 0:
                break
        else:
            raise RuntimeError(
                "Error: No suitable position found, \
                maybe you are trying to insert to many particles? Aborting."
            )

    return _renumber_projectile_resids(TargetUniverse, nAtomsTargetOriginal)

def InsertPlanarFromDistribution(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: float | None = None,
    ymax: float | None = None,
    zmax: float | None = None,
    distance: float = 1.25,
    probability: Callable | None = None,
    fudge_factor: float = 1.2,
    n_tries: int = 1000,
    n_grid_points: int = 1000,
    dim: int = 2,
) -> mda.Universe:

    def insert_ions(
        TargetUniverse,
        ProjectileUniverse,
        InsertionDomain,
        positions,
        n,
        distance,
        n_tries
    ):
        """
        Insert ions into the target universe at specified positions.

        Positional arguments:
        TargetUniverse     -- The universe to insert ions into.
        ProjectileUniverse -- The universe containing the ions to insert.
        positions          -- The positions to insert the ions at.
        distance           -- Minimum distance between inserted ions and existing atoms.
        n_tries            -- Number of attempts to find a valid insertion position.

        Returns:
        Updated TargetUniverse with inserted ions.
        """

        nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
        dimensionsTarget = TargetUniverse.dimensions.copy()
        nAtomsStart = TargetUniverse.atoms.n_atoms

        if TargetUniverse.atoms.n_atoms == 0:
            TargetUniverse = ProjectileUniverse.copy()
            TargetUniverse.dimensions = dimensionsTarget

            t_vec = pos_random(InsertionDomain) - ProjectileUniverse.atoms.center_of_geometry()
            first_z_position = positions[0]
            positions = np.delete(positions, 0) # Remove the first z position as it's already used

            t_vec[dim] = first_z_position - ProjectileUniverse.atoms.center_of_geometry()[dim]
            TargetUniverse.atoms.translate(
                t_vec
            )
            TargetUniverse.atoms.rotateby(*rot_random())


        while True:
            nAtomsTarget = TargetUniverse.atoms.n_atoms

            TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
            TargetUniverse.dimensions = dimensionsTarget

            target = TargetUniverse.atoms[0:nAtomsTarget]
            projectile = TargetUniverse.atoms[-nAtomsProjectile:]
            ns = mda.lib.NeighborSearch.AtomNeighborSearch(target, dimensionsTarget)

            for _attempt in range(n_tries):
                t_vec = pos_random(InsertionDomain) - projectile.atoms.center_of_geometry()
                if len(z_positions) == 0:
                    raise RuntimeError(
                        "Error: No more z positions available for insertion. Increase the fudge factor and try again."
                    )
                next_z = z_positions[0]
                z_positions = np.delete(z_positions, 0) # Remove the first z position as it's already used

                t_vec[dim] = next_z - projectile.atoms.center_of_geometry()[dim]
                projectile.translate(t_vec)
                projectile.rotateby(*rot_random())

                if len(ns.search(projectile, distance)) == 0:
                    break
            else:
                raise RuntimeError(
                    "Error: No suitable position found,\
                    maybe you are trying to insert to many particles? Aborting."
                )

            projectile.residues.resids = (
                projectile.residues.resids + target.residues.resids[-1]
            )
            if TargetUniverse.atoms.n_atoms - nAtomsStart >= n * nAtomsProjectile:
                break
        return TargetUniverse

    # Check TargetUniverse dimensions
    if TargetUniverse.dimensions is None:
        raise ValueError("TargetUniverse must have defined dimensions.")
    if xmax is None:
        xmax = TargetUniverse.dimensions[0]
    if ymax is None:
        ymax = TargetUniverse.dimensions[1]
    if zmax is None:
        zmax = TargetUniverse.dimensions[2]
    if xmin is None:
        xmin = 0
    if ymin is None:
        ymin = 0
    if zmin is None:
        zmin = 0

    # Define insertion domain
    InsertionDomain = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]

    # Draw more than needed ions to account for rejections during insertion
    # NIonsToDraw = np.ceil(n * fudge_factor).astype(int)

    grid = np.linspace(InsertionDomain[dim], InsertionDomain[dim + 3], n_grid_points)
    if probability is None:
        raise ValueError("A probability distribution function must be provided.")
    try:
        p = probability.calculate_p(grid)
    except Exception as exc:
        raise ValueError("The provided probability function is not valid.") from exc

    positionsToDraw = int(np.ceil(n * fudge_factor))

    samples = np.random.choice(len(p), size=positionsToDraw, p=p)
    # Convert indices to z positions
    positions = grid[samples]

    # Insert ions into the TargetUniverse
    TargetUniverse = insert_ions(
        TargetUniverse, ProjectileUniverse, InsertionDomain, positions, n, distance, n_tries)

    return TargetUniverse

def PlanarPoissonBoltzmann(
    TargetUniverse: mda.Universe,
    AnionProjectileUniverse: mda.Universe,
    CationProjectileUniverse: mda.Universe,
    N_anions: int,
    N_cations: int,
    epsilon_r: float = 80.2,
    T: float = 300.0,
    q_diff: int = 0,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: float | None = None,
    ymax: float | None = None,
    zmax: float | None = None,
    dim: int = 2,
    n_grid_points: int = 1000,
    distance: float = 1.25,
    fudge_factor: float = 1.5,
    n_tries: int = 100,
) -> mda.Universe:
    """
    Inserts ions into a system with plate capacitor geometry according to a Poisson-Boltzmann
    distribution. 

    Positional arguments:
    TargetUniverse           -- MDAnalysis Universe of the target system.
    AnionProjectileUniverse  -- MDAnalysis Universe of the anion projectile.
    CationProjectileUniverse -- MDAnalysis Universe of the cation projectile.
    N_anions                 -- Number of anions to insert.
    N_cations                -- Number of cations to insert.

    Keyword arguments:
    epsilon_r                -- Relative permittivity of the dielectricum (dimensionless).
    T                        -- Temperature in Kelvin.
    q_diff                   -- Total charge difference between cations and anions.
    xmin, ymin, zmin         -- Minimum coordinates of the insertion domain.
    xmax, ymax, zmax         -- Maximum coordinates of the insertion domain.
    dim                      -- Dimension orthogonal to the plates (0 = x, 1 = y, 2 = z).
    n_grid_points            -- Number of grid points for calculating the potential profile.
    distance                 -- Minimum distance between inserted ions and existing atoms.
    fudge_factor             -- Fudge factor for number of inserted ions.
    n_tries                  -- Number of attempts to find a valid insertion position.

    Returns:
    Solvated Universe with inserted ions.
    """

    def debye_length(epsilon_r, T, cN_bulk_cat, cN_bulk_an):
        """
        Calculate the Debye length.
        
        Positional arguments:
        epsilon_r   -- Relative permittivity of the dielectricum (dimensionless).
        T           -- Temperature in Kelvin.
        cN_bulk_cat -- Bulk concentration of cations in 1/Å³.
        cN_bulk_an  -- Bulk concentration of anions in 1/Å³.
        
        Returns:
        Debye length.
        """

        return np.sqrt(epsilon_r * epsilon_0 * kB * T / (2 * e**2 * (cN_bulk_cat + cN_bulk_an)/2))

    def electrostatic_potential_profile(sigma, lambda_D, z, epsilon_r):
        """
        Calculate the electrostatic potential profile.
        
        Positional arguments:
        sigma      -- Surface charge density in e/Å².
        lambda_D   -- Debye length in Å.
        z          -- Distance from the charged surface in Å.
        epsilon_r  -- Relative permittivity of the dielectricum (dimensionless).
        
        Returns:
        Electrostatic potential profile in V.
        """

        return sigma / (epsilon_r * epsilon_0 * e) * lambda_D * np.exp(-z / lambda_D)

    def pb_factor_profile(q, T, phi):
        """
        Calculate the Poisson-Boltzmann factor profile.
        
        Positional arguments:
        q    -- Charge of the ion in e.
        T    -- Temperature in K.
        phi  -- Electrostatic potential profile in V.
        
        Returns:
        Poisson-Boltzmann factor profile in .
        """

        p = np.exp(-q * e * phi / (kB * T))
        p /= np.sum(p)
        return p

    def generate_z_positions(pbfp, N, z):
        """
        Generate z positions based on the Poisson-Boltzmann factor.
        
        Positional arguments:
        pbfp -- Poisson-Boltzmann factor profile.
        N  -- Number of ions to place.
        z  -- z positions corresponding to the Boltzmann factor profile.
        
        Returns:
        z positions of the ions.
        """

        pbfp = pbfp.to('dimensionless').magnitude
        samples = np.random.choice(len(pbfp), size=N, p=pbfp)

        # Convert indices to z positions
        z_positions = z[samples]
        return z_positions

    def insert_ions(TargetUniverse, ProjectileUniverse, z_positions, distance, n_tries):
        """
        Insert ions into the target universe at specified z positions.

        Positional arguments:
        TargetUniverse   -- The universe to insert ions into.
        ProjectileUniverse -- The universe containing the ions to insert.
        z_positions       -- The z positions to insert the ions at.
        distance          -- Minimum distance between inserted ions and existing atoms.
        n_tries             -- Number of attempts to find a valid insertion position.

        Returns:
        Updated TargetUniverse with inserted ions.
        """

        nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

        # No pint units for MDAnalysis
        distance = distance.to('angstrom').magnitude
        z_positions = z_positions.to('angstrom').magnitude

        if TargetUniverse.atoms.n_atoms == 0:
            TargetUniverse = ProjectileUniverse.copy()
            TargetUniverse.dimensions = dimensionsTarget

            t_vec = pos_random(InsertionDomain) - ProjectileUniverse.atoms.center_of_geometry()
            first_z_position = z_positions[0]
            z_positions = np.delete(z_positions, 0)


            t_vec[2] = first_z_position - ProjectileUniverse.atoms.center_of_geometry()[2]
            TargetUniverse.atoms.translate(
                t_vec
            )
            TargetUniverse.atoms.rotateby(*rot_random())

        for _N, z in tqdm(enumerate(z_positions)):
            nAtomsTarget = TargetUniverse.atoms.n_atoms

            TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
            TargetUniverse.dimensions = dimensionsTarget

            target = TargetUniverse.atoms[0:nAtomsTarget]
            projectile = TargetUniverse.atoms[-nAtomsProjectile:]
            ns = mda.lib.NeighborSearch.AtomNeighborSearch(target, dimensionsTarget)

            for _attempt in range(n_tries):
                t_vec = pos_random(InsertionDomain) - projectile.atoms.center_of_geometry()
                t_vec[2] = z - projectile.atoms.center_of_geometry()[2]
                projectile.translate(
                    t_vec
                )

                projectile.rotateby(*rot_random())

                if len(ns.search(projectile, distance)) == 0:
                    break
            else:
                raise RuntimeError(
                    "Error: No suitable position found,\
                    maybe you are trying to insert to many particles? Aborting."
                )

            projectile.residues.resids = (
                projectile.residues.resids + target.residues.resids[-1]
            )
        return TargetUniverse

    q_excess = N_cations - N_anions # elementary charge
    print(f"Total excess charge to be compensated: {q_excess} e")

    # Calculate plate charges
    if q_excess != 0:
        q_1 = (-q_diff + q_excess) / 2
        q_2 = (q_diff + q_excess) / 2
    else:
        q_1 = -q_diff / 2
        q_2 = q_diff / 2

    # Check TargetUniverse dimensions
    if TargetUniverse.dimensions is None:
        raise ValueError("TargetUniverse must have defined dimensions.")
    if xmax is None:
        xmax = TargetUniverse.dimensions[0]
    if ymax is None:
        ymax = TargetUniverse.dimensions[1]
    if zmax is None:
        zmax = TargetUniverse.dimensions[2]
    if xmin is None:
        xmin = 0
    if ymin is None:
        ymin = 0
    if zmin is None:
        zmin = 0

    # Define insertion domain
    InsertionDomain = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomainSize = InsertionDomain[3:6] - InsertionDomain[0:3] # angstrom
    dimensionsTarget = TargetUniverse.dimensions.copy()

    # Calculate surface charge densities
    area = np.prod(np.delete(InsertionDomainSize, dim)) # angstrom^2
    sigma_1 = q_1 / area
    sigma_2 = q_2 / area
    print(f"Surface charge density 1: {sigma_1} e/Å²")
    print(f"Surface charge density 2: {sigma_2} e/Å²")

    # Calculate bulk concentration
    volume = np.prod(InsertionDomainSize) # angstrom^3
    cN_bulk_cat = N_cations / volume
    cN_bulk_an  = N_anions / volume

    # Grid in dim-direction
    grid = np.linspace(0, InsertionDomainSize[dim], n_grid_points)

    # Calculate Debye length
    l_D = debye_length(epsilon_r, T, cN_bulk_cat, cN_bulk_an)
    print(f"Debye length : {l_D:.2f} Å")

    # Calculate potential profiles
    phi_1 = electrostatic_potential_profile(sigma_1, l_D, grid, epsilon_r)
    phi_2 = electrostatic_potential_profile(sigma_2, l_D, InsertionDomainSize[dim] - grid, epsilon_r)
    phi_total = phi_1 + phi_2
    print(f'Potential difference between plates: {
        (phi_total[0] - phi_total[-1]):.2f} V')

    # Calculate Poisson-Boltzmann factor profiles
    pbfp_anions = pb_factor_profile(-e, T, phi_total)
    pbfp_cations = pb_factor_profile(e, T, phi_total)

    # Draw more than needed ions to account for rejections during insertion
    N_anions_to_draw = np.ceil(N_anions * fudge_factor).astype(int)
    N_cations_to_draw = np.ceil(N_cations * fudge_factor).astype(int)

    # Generate z positions based on Poisson-Boltzmann distribution
    positions_anions = generate_z_positions(pbfp_anions, N_anions_to_draw, grid)
    positions_cations = generate_z_positions(pbfp_cations, N_cations_to_draw, grid)

    # Remove positions that are too close to the plates
    positions_anions = positions_anions[
        (positions_anions > distance) &
        (positions_anions < (InsertionDomainSize[dim] - distance))
    ]
    positions_cations = positions_cations[
        (positions_cations > distance) &
        (positions_cations < (InsertionDomainSize[dim] - distance))
    ]

    # Select only the required number of ions
    positions_anions = positions_anions[0:N_anions]
    positions_cations = positions_cations[0:N_cations]

    print(len(positions_anions), "anions to be inserted.")
    print(len(positions_cations), "cations to be inserted.")

    # Adjust z positions to absolute coordinates (to be improved)
    positions_anions = positions_anions + zmin
    positions_cations = positions_cations + zmin

    # Insert ions into the TargetUniverse
    TargetUniverse = insert_ions(
        TargetUniverse, AnionProjectileUniverse, positions_anions, distance, n_tries)
    TargetUniverse = insert_ions(
        TargetUniverse, CationProjectileUniverse, positions_cations, distance, n_tries)

    return TargetUniverse
