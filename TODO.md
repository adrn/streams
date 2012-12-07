TODO
====
 * Rewrite tests in test_potential.py to be like test_point_mass_3d()
 * Maybe what I should do is look for times when (r < r_tide) & (v < v_esc)?
 * The times are screwing me up here -- is the snapshot from the last time in the sgr_cen file? When I do the backwards
     integration, I integrate from 0 to -6000 Myr, but I think what I actually want to do is integrate from 6000 to 0,
     but with opposite velocity? Or does the negative time step take care of that?
 * potential.plot() should accept axes and do the right thing when I over-plot on it