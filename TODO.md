TODO
====
 * Rewrite tests in test_potential.py to be like test_point_mass_3d()
 * potential.plot() should accept axes and do the right thing when I over-plot on it
 * Clean up plotting in both back_integrate files
 * Papers to read:
     * Longmore, Fernley & Jameson (1986, MNRAS, 220, 279)
     * Madore & Freedman (2011, ApJ, 744, 132)
 * Email Carl Grillmair Distance vs. RA for the nearest wrap of the SGR simulation data from Law & Majewski 2010
 * Check Catalina and PTF for RR Lyrae in TriAnd -> Email Andrew Drake

2013-02-06
 * Make some proper motion plot
 * See if A. Drake has a table of distance, l, b, ra, dec
 * Compare how we can do with just Pan-STARRs (2 wraps) vs. those data + a wrap further in orbital phase

2013-02-08
 * We don't have a generative model for our data -- e.g. the likelihood function I wrote down is
     some heuristic that I pulled out of my ass. The right way to do it would be to write down a
     model that can reproduce the density of observed data points along the stream -- Dan mentioned
     doing this and using a KDE + a convolution.
 * Really I just wrote down an arbitrary scalar objective function that I want to minimize, so using
     MCMC is probably wrong -- I can just do an optimization and Monte Carlo dat shit with observational
     errors?