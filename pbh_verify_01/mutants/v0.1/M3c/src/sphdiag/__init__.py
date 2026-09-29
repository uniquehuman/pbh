"""sphdiag -- physical diagnostics of spherically symmetric 3+1 slices (PBH-VERIFY-01).

Implements physics_contract.md v1.0 §10:

    SphericalSlice, ScalarField, VlasovMoments, Projections, pi_from_dtphi, diagnose
"""
from .diagnose import (DIAGNOSTICS_FIELDS, GEOMETRY_ORDER, Diagnostics, Sign, diagnose)
from .fd import RadialOperator, fornberg_weights
from .matter import Projections, ScalarField, VlasovMoments, pi_from_dtphi
from .slice import SphericalSlice

__version__ = "1.0.0"
CONTRACT_VERSION = "1.0"

__all__ = [
    "SphericalSlice", "ScalarField", "VlasovMoments", "Projections", "pi_from_dtphi",
    "diagnose", "Diagnostics", "Sign", "DIAGNOSTICS_FIELDS", "GEOMETRY_ORDER",
    "RadialOperator", "fornberg_weights", "CONTRACT_VERSION",
]
