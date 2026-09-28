"""sphdiag: physical diagnostics of spherically symmetric 3+1 slices (physics_contract.md v1.0).

G = c = 1, signature (-,+,+,+).  Public API (contract section 10):
    SphericalSlice, ScalarField, VlasovMoments, Projections, pi_from_dtphi, diagnose, Diagnostics
"""
from .slice import SphericalSlice
from .matter import ScalarField, VlasovMoments, Projections, pi_from_dtphi
from .diagnose import diagnose, Diagnostics, Sign

__all__ = ["SphericalSlice", "ScalarField", "VlasovMoments", "Projections", "pi_from_dtphi",
           "diagnose", "Diagnostics", "Sign"]
__version__ = "1.0.0"
