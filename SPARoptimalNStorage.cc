#include <octave/oct.h>
#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>
#include <octave/ov-struct.h>
#include <octave/intNDArray.h>
#include <iostream>
#include <QuadProg++.hh>

using namespace std;

struct opt_sol
{
	float F;
	FloatNDArray xc;
	FloatNDArray xd;
};

int32NDArray numR;
int numS;
int numSfin;
int numW;
int numN;
glp_prob *lp;
glp_smcp parm_lp;
glp_iocp parm_mip;
float T;
float rho;

// Storage parameters
FloatNDArray Qmax;
FloatNDArray Qmin;
FloatNDArray q0;
FloatNDArray C;
FloatNDArray D;
FloatNDArray nul;
FloatNDArray nuc;
FloatNDArray nud;
FloatNDArray DeltaCmax;
FloatNDArray DeltaDmax;

int32NDArray R;
int32NDArray Rx;
int32NDArray xc;
int32NDArray xd;

int32NDArray set_fin;

float gama;

// Own header files.
#include "init.h"
#include "linprog.h"
#include "slopeupdate.h"
#include "support.h"

/* ----------------------------------------------------------------------------*
 * DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T")   *
 * ----------------------------------------------------------------------------*
 * Main function. Call "SPARoptimalNStorage(rho, g, r, P, S, numI, T)" in 
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

DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T")
{
	octave_value_list retval;
	int nargin = args.length();
	if (nargin != 7)
		print_usage();
	else
	{
		rho = args(0).float_value();
		FloatNDArray g = args(1).array_value();
		FloatNDArray r = args(2).array_value();
		octave_scalar_map P = args(3).scalar_map_value();
		octave_scalar_map S = args(4).scalar_map_value();
		int numI = args(5).int_value();
		T = args(6).float_value();
		
		// Prices
		FloatNDArray pg = P.contents("pg").array_value();
		FloatNDArray pr = P.contents("pr").array_value();
		FloatNDArray pc = P.contents("pc").array_value();
		FloatNDArray pd = P.contents("pd").array_value();
		
		numN = g.dims().elem(0);
		numW = g.dims().elem(1);
		
		init(S);
		initOpt();
		
		dim_vector dv_v(numR.sum(0).elem(0), numN);
		
		FloatNDArray v(dv_v, 0); // Value function for the different levels
		int32NDArray NV(dv_v, 0); // Number of visits to the corresponding state
		opt_sol ret;
		FloatNDArray vhat;
		FloatNDArray z;
		int32NDArray smpl;
		
		FloatNDArray q(dim_vector(numSfin, numN));
		FloatNDArray uc(dim_vector(numS, numN));
		FloatNDArray ud(dim_vector(numS, numN));
		FloatNDArray cost(dim_vector(numN, numI), 0);
		gama = 0.95;
		octave_int32 index;
		
		FloatNDArray alpha(dim_vector(1, numN));
		float nu = 0.2;
		float alpha0 = 10;
		FloatNDArray lambda(dim_vector(1, numN), pow(alpha0, 2));
		FloatNDArray delta(dim_vector(1, numN), alpha0);
		FloatNDArray c(dim_vector(1, numN), 1);
		FloatNDArray sigma2(dim_vector(1, numN), 1);
		
		for (int i=0; i<numI; i++)
		{
			smpl = randi(0,numW,numN); // Generate sample
			
			for (int k=0; k<numN; k++)
			{
				if (k != 0) 
				{
					// Compute pre-decision asset level
					R.insert(Rx.column(k-1), 0, k);
					
					// Find optimal value function and 
					// compute post-decision asset level
					ret = solveOpt(
						g.column(smpl(k)).elem(k), r.column(smpl(k)).elem(k),
						pc.page(smpl(k)).column(k), pd.page(smpl(k)).column(k),
						R.column(k), v.column(k),
						xc.column(k-1), xd.column(k-1)
					);
				}
				else
				{
					// Find optimal value function and 
					// compute post-decision asset level
					ret = solveOpt(
						g.column(smpl(k)).elem(k), r.column(smpl(k)).elem(k),
						pc.page(smpl(k)).column(k), pd.page(smpl(k)).column(k),
						R.column(k), v.column(k),
						int32NDArray(dim_vector(numS, 1), 0), int32NDArray(dim_vector(numS, 1), 0)
					);
				}
				
				xc.insert(ret.xc, 0, k);
				xd.insert(ret.xd, 0, k);
				
				float Rxerr;
				int count = 0;
				for (int m=0; m<numS; m++)
				{
					if ((int)set_fin(m) == 1)
					{
						// Resource transition function
						Rxerr = nul(m)*(float)R(count, k)+T*(
							nuc(m)*(float)xc(m, k)-
							(1/nud(m))*(float)xd(m, k)
						);
						
						if (Rxerr < rho*Qmin(m))
						{
							Rx(count, k) = floor(rho*Qmin(m));
						}
						else if (Rxerr > rho*Qmax(m))
						{
							Rx(count, k) = floor(rho*Qmax(m));
						}
						else Rx(count, k) = floor(Rxerr);
						
						count = count+1;
					}
				}
				
				// Update step
				c(k) = (1-nu)*c(k)+nu*ret.F;
				sigma2(k) = (1-nu)*sigma2(k)+nu*pow(c(k)-ret.F, 2);
				
				alpha(k) = ((1-gama)*lambda(k)*sigma2(k)+pow(1-(1-gama)*delta(k), 2)*pow(c(k), 2))/
					(pow(1-gama, 2)*lambda(k)*sigma2(k)+pow(1-(1-gama)*delta(k), 2)*pow(c(k), 2)+sigma2(k));
				
				lambda(k) = pow(alpha(k), 2)+pow(1-(1-gama)*alpha(k), 2)*lambda(k);
				delta(k) = alpha(k)+(1-(1-gama)*alpha(k))*delta(k);
				
//				octave_int32 index = 0;
//				for (int m=0; m<numSfin; m++)
//				{
//					NV(index+Rx(m, k), k) = NV(index+Rx(m, k), k)+(octave_int32)1;
//					alpha(k) = 1/(float)NV(index+Rx(m, k), k);
//					index = index+numR(m);
//				}
				
				if (k < numN-1)
				{
					// Observe slope
					vhat = observeslope(g(k+1, smpl(k+1)), r(k+1, smpl(k+1)),
						pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
						Rx.column(k), v.column(k+1), xc.column(k), xd.column(k)
					);
					
					index = 0;
					for (int m=0; m<numSfin; m++)
					{
						// Update slope
						z = updateslope(
							v.column(k).linear_slice(index, index+numR(m)),
							vhat.column(m), alpha(k), Rx(m, k)
						);
						
						// Project slope
						v.insert(projectslope(z, Rx(m, k)), index, k);
						
						index = index+numR(m);
					} // endfor m
				} // endif
			} // endfor k
			
			for (int k=0; k<numN; k++)
			{
				for (int m=0; m<numS; m++)
				{
					uc(m, k) = xc(m, k)/rho;
					ud(m, k) = xd(m, k)/rho;
					
					cost(k, i) = cost(k, i)+uc(m, k)*pc(m, k)+ud(m, k)*pd(m, k);
				}
				cost(k, i) = cost(k, i)+g(k)*pg(k)+r(k)*pr(k);
			}
			
		} // endfor iter
		
		// Rescale return value
		
		for (int k=0; k<numN; k++)
		{
			for (int m=0; m<numSfin; m++)
			{
				q(m, k) = (float)Rx(m, k)/rho;
			}
		}
		
		retval(0) = octave_value(q);
		retval(1) = octave_value(uc);
		retval(2) = octave_value(ud);
		retval(3) = octave_value(cost);
		retval(4) = octave_value(v);
		
		deleteOpt();
	}
return retval;
}
