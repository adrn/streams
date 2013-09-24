def variance_projections(**kwargs):
    """ Figure showing 2D projections of the 6D variance """
    
    params = true_params.copy()
    params['qz'] = true_params['qz']*1.2
    #params['v_halo'] = true_params['v_halo']*1.2
    
    # define both potentials
    correct_lm10 = LawMajewski2010(**true_params)
    wrong_lm10 = LawMajewski2010(**params)
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'axes.facecolor' : '#ffffff'}
    
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(2, 2, figsize=(10,10))
        colors = ['#0571B0', '#D7191C']
        markers = ['o', '^']
        labels = ['true', '20% wrong $q_z$']
        for ii,potential in enumerate([correct_lm10, wrong_lm10]):
            integrator = SatelliteParticleIntegrator(potential, satellite, particles)
            s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
            min_ps = minimum_distance_matrix(potential, s_orbit, p_orbits)
            
            axes[0,0].plot(min_ps[:,0], min_ps[:,3], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii], label=labels[ii])
            axes[1,0].plot(min_ps[:,0], min_ps[:,4], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii])
            axes[0,1].plot(min_ps[:,1], min_ps[:,3], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii])
            axes[1,1].plot(min_ps[:,1], min_ps[:,4], markers[ii], alpha=0.75, 
                           linestyle='none', color=colors[ii])
        
        # limits
        for ax in np.ravel(axes):
            ax.set_xlim(-4,4)
            ax.set_ylim(-4,4)
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            ax.xaxis.set_ticks([-2,0,2])
            ax.yaxis.set_ticks([-2,0,2])
        
        # tick hacking
        axes[0,0].set_xticks([])
        axes[0,1].set_xticks([])
        axes[0,1].set_yticks([])
        axes[1,1].set_yticks([])
        
        # labels
        axes[0,0].set_ylabel(r'$p_x$')
        axes[1,0].set_xlabel(r'$q_x$')
        axes[1,0].set_ylabel(r'$p_y$')
        axes[1,1].set_xlabel(r'$q_y$')
        
        axes[0,0].legend(loc='upper left', fancybox=True)
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    #plt.show()
    fig.savefig(os.path.join(plot_path, "variance_projections.pdf"))