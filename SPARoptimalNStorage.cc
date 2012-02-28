#include <octave/oct.h>
#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>
#include <octave/ov-struct.h>
#include <octave/intNDArray.h>
#include <iostream>
#include <QuadProg++.hh>

using namespace std;

int32NDArray numR;
int numS;
int numSfin;
int numW;
int numN;
glp_prob *lp;
glp_smcp parm_lp;
float T;
float rho;

// Storage parameters
FloatNDArray Qmax;
FloatNDArray Qmin;
FloatNDArray q0;
FloatNDArray C;
FloatNDArray D;
FloatNDArray etal;
FloatNDArray etac;
FloatNDArray etad;
FloatNDArray DeltaCmax;
FloatNDArray DeltaDmax;

int32NDArray set_fin;

float gama = 0.5; // 0 <= gamma <= 1 0.5
float alpha0 = 0.8; // 0 <= alpha0 <= 1 0.8
float deltaStepMult = 0.8; // 0.8
float a = 4; // 4
float b = 100; // 40
float c = 0.5; // 0.4
//const float nu = 0.2;
//const float c0 = 1;
//const float sigma20 = 1;

// Own source files.
#include "init.h"
#include "linprog.h"
#include "transition.h"
#include "slopeupdate.h"
#include "utils.h"

/* ----------------------------------------------------------------------------*
 * DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T,    *
 *  parm")                                                                     *
 * ----------------------------------------------------------------------------*
 * Main function. Call "SPARoptimalNStorage(rho, g, r, P, S, numI, T, parm)" in 
 * Octave.
 *
 * @param rho Scaling factor for the quantisation.
 * @param g numW samples of generated energy for every time step numN.
 * @param r numW samples of requested energy for every time step numN.
 * @param P Structure with numW samples of energy cost for every time step numN.
 * @param S Structure with storage parameters.
 * @param numI Number of Iterations.
 * @param T Time step size.
 *
 * @return q, uc, ud, cost
 *
 */ 

DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T, parm")
{
	octave_value_list retval;
	
	if (args.length() < 7) print_usage();
	else
	{
		rho = args(0).float_value();
		FloatNDArray g = args(1).array_value();
		FloatNDArray r = args(2).array_value();
		octave_scalar_map P = args(3).scalar_map_value();
		octave_scalar_map S = args(4).scalar_map_value();
		int numI = args(5).int_value();
		T = args(6).float_value();
		octave_scalar_map parm = args(7).scalar_map_value();
		
		// Parameters
		gama = parm.contents("gamma").float_value();
		alpha0 = parm.contents("alpha0").float_value();
		deltaStepMult = parm.contents("deltaStepMult").float_value();
		a = parm.contents("a").float_value();
		b = parm.contents("b").float_value();
		c = parm.contents("c").float_value();
		float epsilon = parm.contents("epsilon").float_value();
		
		// Prices
		FloatNDArray pg = P.contents("pg").array_value();
		FloatNDArray pr = P.contents("pr").array_value();
		FloatNDArray pc = P.contents("pc").array_value();
		FloatNDArray pd = P.contents("pd").array_value();
		
		numN = g.dims().elem(0);
		numW = g.dims().elem(1);
		
		// Pre-decision asset level
		int32NDArray R = init(S);
		
		// Post-decision asset level
		int32NDArray Rx = int32NDArray(dim_vector(numSfin, numN));
		
		// Value function for the different levels
		FloatNDArray v(dim_vector(numR.sum(0).elem(0), numN), 0);
		
		opt_sol ret;
		FloatNDArray vhat;
		FloatNDArray z;
		int32NDArray smpl;
		FloatNDArray xc = FloatNDArray(dim_vector(numS, numN), 0);
		FloatNDArray xd = FloatNDArray(dim_vector(numS, numN), 0);
		
		FloatNDArray alpha(dim_vector(1, numN));
//		FloatNDArray lambda(dim_vector(1, numN), pow(alpha0, 2));
//		FloatNDArray delta(dim_vector(1, numN), alpha0);
//		FloatNDArray c(dim_vector(1, numN), c0); 
//		FloatNDArray sigma2(dim_vector(1, numN), sigma20);
		FloatNDArray deltaStep(dim_vector(numSfin, numN));
		for (int m=0; m<numSfin; m++)
		{
			for (int k=0; k<numN; k++)
			{
				deltaStep(m, k) = deltaStepMult*(float)numR(m);
			}
		}
		
//		FloatNDArray S1(dim_vector(numN, 1), 0);
//		FloatNDArray S2(dim_vector(numN, 1), 0);
//		FloatNDArray e(dim_vector(numN, 1), 0);
		
		// Return values
		FloatNDArray cost(dim_vector(numN, numI), 0);
		
		initLinProg();
		
		for (int i=1; i<numI; i++)
		{
			smpl = randi(0, numW, numN); // Generate sample
			
			for (int k=0; k<numN; k++)
			{
				// Compute pre-decision asset level
				if (k > 0) R.insert(Rx.column(k-1), 0, k);
				
				// Find optimal value function and 
				// compute post-decision asset level
				ret = solveLinProg(
					g(k, smpl(k)), r(k, smpl(k)),
					pc.page(smpl(k)).column(k), pd.page(smpl(k)).column(k),
					R.column(k), v.column(k),
					xc.column(k), xd.column(k)
				);
				
				xc.insert(ret.xc, 0, k);
				xd.insert(ret.xd, 0, k);
				
				// Resource transition function
				Rx.insert(transitionResource(R.column(k), xc.column(k), xd.column(k)), 0, k);
				
				cost(k, i) = g(k, smpl(k))*pg(k, smpl(k))+r(k, smpl(k))*pr(k, smpl(k));
				for (int m=0; m<numS; m++)
				{
					cost(k, i) = cost(k, i)+xc(m, k)/rho*pc(m, k, smpl(k))+xd(m, k)/rho*pd(m, k, smpl(k));
				}
				
				// Update step
//				alpha(k) = ((1-gama)*lambda(k)*sigma2(k)+pow(1-(1-gama)*delta(k), 2)*pow(c(k), 2))/
//					(pow(1-gama, 2)*lambda(k)*sigma2(k)+pow(1-(1-gama)*delta(k), 2)*pow(c(k), 2)+sigma2(k));
//				c(k) = (1-nu)*c(k)+nu*cost(k, i);
//				sigma2(k) = (1-nu)*sigma2(k)+nu*pow(c(k)-cost(k, i), 2);
//				lambda(k) = pow(alpha(k), 2)+pow(1-(1-gama)*alpha(k), 2)*lambda(k);
//				delta(k) = alpha(k)+(1-(1-gama)*alpha(k))*delta(k);
				
				alpha(k) = alpha0*(b/i+a)/(b/i+a+pow(i, c));
				
				if (k < numN-1)
				{
					// Observe slope
					vhat = observeSlope(g(k+1, smpl(k+1)), r(k+1, smpl(k+1)),
						pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
						Rx.column(k), v.column(k+1), xc.column(k+1), xd.column(k+1)
					);
					
//					e(k) = v(Rx(0, k), k)-vhat(0);
//					S1(k) = (1-0.9)*S1(k)+0.9*e(k);
//					S2(k) = (1-0.9)*S2(k)+0.9*abs(e(k));
//					alpha(k) = abs(S1(k))/S2(k);
					
					octave_int32 index = 0;
					for (int m=0; m<numSfin; m++)
					{
						// Update slope
						z = updateSlope(
							v.column(k).linear_slice(index, index+numR(m)),
							vhat(m), alpha(k), Rx(m, k), deltaStep(m, k)
						);
						
						// Project slope
						v.insert(projectSlope(z, Rx(m, k)), index, k);
						
						index = index+numR(m);
					} // endfor m
				} // endif
			} // endfor k
			
			for (int m=0; m<numSfin; m++)
			{
				for (int k=0; k<numN; k++)
				{
					if (cost(k, i) >= cost(k, i-1)+epsilon)
					{
						deltaStep(m, k) = fmax(0, 0.5*deltaStep(m, k));
					}
				}
			}
			
		} // endfor iter
		
		// Rescale return value
		retval(0) = octave_value((FloatNDArray)Rx/rho);
		retval(1) = octave_value(xc/rho);
		retval(2) = octave_value(xd/rho);
		retval(3) = octave_value(cost.column(numI-1));
		retval(4) = octave_value(cost);
		
		deleteLinProg();
	}
	
	return retval;
}
