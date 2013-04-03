# coding: utf-8
"""
    Test the prior / likelihood functions for back integration.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest

from ..inference import make_prior

def test_make_prior():
    ln_prior = make_prior(["qz", "phi", "v_halo"])
    
    assert np.isinf(ln_prior([-1.,2.,0.14]))
    assert np.isfinite(ln_prior([1.4,2.,0.14]))
    
    with pytest.raises(IndexError):
        ln_prior([1.4,2.])