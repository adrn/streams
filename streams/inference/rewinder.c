#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Won't work if you want to parallelize!
const double Rsun = 8.;
// double R, latitude, log_jac;
// double x1, x2, x3, v1, v2, v3;
// double r_term, v_term;

double dot3D(double *a, double *b) {
    /*
        Compute a dot product in 3 dimensions.
    */
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void set_basis(double progenitor_x[], double progenitor_v[],
               double x1_hat[], double x2_hat[], double x3_hat[]) {
    /*
        Cartesian basis defined by the orbital plane of the satellite to
        project the orbits of stars into.
    */

    double x1_norm, x2_norm, x3_norm = 0.;

    x1_hat[0] = progenitor_x[0];
    x1_hat[1] = progenitor_x[1];
    x1_hat[2] = progenitor_x[2];

    x3_hat[0] = x1_hat[1]*progenitor_v[2] - x1_hat[2]*progenitor_v[1];
    x3_hat[1] = x1_hat[2]*progenitor_v[0] - x1_hat[0]*progenitor_v[2];
    x3_hat[2] = x1_hat[0]*progenitor_v[1] - x1_hat[1]*progenitor_v[0];

    x2_hat[0] = -x1_hat[1]*x3_hat[2] + x1_hat[2]*x3_hat[1];
    x2_hat[1] = -x1_hat[2]*x3_hat[0] + x1_hat[0]*x3_hat[2];
    x2_hat[2] = -x1_hat[0]*x3_hat[1] + x1_hat[1]*x3_hat[0];

    x1_norm = sqrt(dot3D(x1_hat, x1_hat));
    x2_norm = sqrt(dot3D(x2_hat, x2_hat));
    x3_norm = sqrt(dot3D(x3_hat, x3_hat));

    for (int i=0; i < 3; i++) {
        x1_hat[i] /= x1_norm;
        x2_hat[i] /= x2_norm;
        x3_hat[i] /= x3_norm;
    }
}

double test(double (*x)[3], long nparticles) {
    double summ = 0.;

    for (int i=0; i < nparticles; i++) {
        for (int j=0; j < 3; j++) {
            summ += x[i][j];
        }
    }

    return summ;
}

void ln_likelihood_helper(double rtide, double vdisp,
                          double prog_x[], double prog_v[],
                          double star_x[][3], double star_v[][3], long nparticles,
                          double x1_hat[], double x2_hat[], double x3_hat[],
                          double dx[], double dv[],
                          double alpha, double betas[],
                          double ln_likelihoods[]) {

    // Define constants for this function
    const double r_norm = -log(rtide) - 0.91893853320467267;
    const double v_norm = -log(vdisp) - 0.91893853320467267;
    const double sigma_r_sq = rtide*rtide;
    const double sigma_v_sq = vdisp*vdisp;

    // Iteration variables
    int i, k;

    // For Jacobian (spherical -> cartesian)
    double R2, latitude, log_jac;

    // Coordinates of stars in instantaneous orbital plane
    double x1, x2, x3, v1, v2, v3;

    // Likelihood terms
    double r_term, v_term;

    // Compute basis defined by instantaneous position of progenitor
    set_basis(prog_x, prog_v, x1_hat, x2_hat, x3_hat);

    for (i=0; i < nparticles; i++) {
        // Translate to be centered on progenitor
        for (k=0; k < 3; k++) {
            dx[k] = star_x[i][k] - prog_x[k];
            dv[k] = star_v[i][k] - prog_v[k];
        }

        // Hijacking these variabless to use for Jacobian calculation
        x1 = star_x[i][0] + Rsun;
        R2 = x1*x1 + star_x[i][1]*star_x[i][1] + star_x[i][2]*star_x[i][2];
        x2 = star_x[i][2]*star_x[i][2]/R2;
        log_jac = log(R2*R2*sqrt(1.-x2));

        // // Project into new basis
        x1 = dx[0]*x1_hat[0] + dx[1]*x1_hat[1] + dx[2]*x1_hat[2];
        x2 = dx[0]*x2_hat[0] + dx[1]*x2_hat[1] + dx[2]*x2_hat[2];
        x3 = dx[0]*x3_hat[0] + dx[1]*x3_hat[1] + dx[2]*x3_hat[2];

        v1 = dv[0]*x1_hat[0] + dv[1]*x1_hat[1] + dv[2]*x1_hat[2];
        v2 = dv[0]*x2_hat[0] + dv[1]*x2_hat[1] + dv[2]*x2_hat[2];
        v3 = dv[0]*x3_hat[0] + dv[1]*x3_hat[1] + dv[2]*x3_hat[2];

        // Move to center of Lagrange point
        // TODO: handle rotation by some angle, theta, in the orbital plane
        x1 -= alpha*betas[i]*rtide;

        // Compute likelihoods for position and velocity terms
        r_term = r_norm - 0.5*(x1*x1 + x2*x2 + x3*x3)/sigma_r_sq;
        v_term = v_norm - 0.5*(v1*v1 + v2*v2 + v3*v3)/sigma_v_sq;

        ln_likelihoods[i] = r_term + v_term + log_jac;

    }
}

