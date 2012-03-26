// Dependencies
//#include "linprog.h"

FloatNDArray observeSlopeDual(float g, float r,
	FloatNDArray pc, FloatNDArray pd, 
	int32NDArray Rx, FloatNDArray v,
	FloatNDArray xc, FloatNDArray xd)
{
	opt_sol retce = solveLinProg(g, r, pc, pd, Rx, v, xc, xd);
	
	return retce.vhat;
}

FloatNDArray observeSlopeDerivative(float g, float r,
	FloatNDArray pc, FloatNDArray pd, 
	int32NDArray Rx, FloatNDArray v,
	FloatNDArray xc, FloatNDArray xd)
{
	FloatNDArray vhat(dim_vector(numSfin, 1));
	opt_sol retup, retlo, retce;
	int32NDArray Rxlo(Rx);
	int32NDArray Rxup(Rx);
	
	retce = solveLinProg(g, r, pc, pd, Rx, v, xc, xd);
	
	for (int m=0; m<numSfin; m++)
	{
		if ((int)Rx(m)+1 == (int)v.dim1())
		{
			// There's no value in adding more than the maximum
			vhat(m) = (octave_int32)0;
		}
		else
		{
			Rxup(m) = Rx(m)+(octave_int32)1;
			
			retup = solveLinProg(g, r, pc, pd, Rxup, v, xc, xd);
			
			vhat(m) = retup.F-retce.F;
		}
	}
	
//	int count = 0;
//	for (int m=0; m<numS; m++)
//	{
//		if ((int)set_fin(m) == 1)
//		{
//			if (Rx(count) == floor(rho*Qmin(m)))
//			{
//				// There's no value in adding more than the maximum
//				vhat(count) = retce.F;//(octave_int32)0;
//			}
//			else
//			{
//				Rxlo(count) = Rx(count)-(octave_int32)1;
//				
//				retlo = solveLinProg(g, r, pc, pd, Rxlo, v, xc, xd);
//				
//				vhat(count) = retce.F-retlo.F;
//			}
//			
//			count = count+1;
//		}
//	}
	
	return vhat;
}

/* ----------------------------------------------------------------------------*
 * FloatNDArray observeSlope(float g, float r, FloatNDArray pc,                *
 *  FloatNDArray pd, int32NDArray Rx, FloatNDArray v,                          *
 *  FloatNDArray xc, FloatNDArray xd)                                          *
 * ----------------------------------------------------------------------------*
 * Observe the slopes.
 *
 * @param g
 * @param r
 * @param pc
 * @param pd
 * @param Rx
 * @param v
 * @param xc
 * @param xd
 *
 * @return vhat
 *
 */

FloatNDArray observeSlope(float g, float r,
	FloatNDArray pc, FloatNDArray pd, 
	int32NDArray Rx, FloatNDArray v,
	FloatNDArray xc, FloatNDArray xd)
{
	return observeSlopeDual(g, r, pc, pd, Rx, v, xc, xd);
}

/* ----------------------------------------------------------------------------*
 * FloatNDArray updateSlope(FloatNDArray v, float vhat, float alpha,    *
 *  octave_int32 Rx)                                                           *
 * ----------------------------------------------------------------------------*
 * Update the vector z with v.
 *
 * @param v
 * @param vhat
 * @param alpha
 * @param Rx
 *
 * @return z
 *
 */

FloatNDArray updateSlope(FloatNDArray v, float vhat, float alpha,
	octave_int32 Rx, float delta)
{
	FloatNDArray z(v);
	
	int lower = floor(fmax((float)Rx-delta, 0));
	int upper = floor(fmin((float)Rx+delta, (float)v.dim1()-1));
	
	for (int m=lower; m<=upper; m++)
	{
		z(m) = (1-(1-gama)*alpha)*v(m)+alpha*vhat;
	}
	
	return z;
}

FloatNDArray projectSlopeLeveling(FloatNDArray z, octave_int32 Rx, float delta)
{
	FloatNDArray v(z);
	int numZ = (int)z.dim1();
	
	int lower = floor(fmax((float)Rx-1-delta, 0));
	int upper = floor(fmin((float)Rx+1+delta, numZ));
	
	for (int r=upper; r<numZ; r++)
	{
		if (v(r) < z(Rx))
		{
			v(r) = z(Rx);
		}
		else break;
	}
	
	for (int r=lower; r>=0; r--)
	{
		if (v(r) > z(Rx))
		{
			v(r) = z(Rx);
		}
		else break;
	}
	
	return v;
}

FloatNDArray projectSlopeMeanLeveling(FloatNDArray z, octave_int32 Rx, float delta)
{
	FloatNDArray v(z);
	int numZ = (int)z.dim1();
	int left = 0;
	int right = 0;
	float val_right = z(Rx)*2*delta;
	float val_left = z(Rx)*2*delta;
	
	int lower = floor(fmax((float)Rx-1-delta, 0));
	int upper = floor(fmin((float)Rx+1+delta, numZ));
	
	for (int r=upper; r<numZ; r++)
	{
		if (v(r) < z(Rx))
		{
			right = right+1;
			val_right = val_right+v(r);
		}
		else break;
	}
	
	if (right > 0)
	{
		FloatNDArray dum(dim_vector(2*delta+right+1, 1), val_right/(right+2*delta));
		v.insert(dum, floor(fmax((float)Rx-delta, 0)), 0);
	}
	else
	{
		for (int r=lower; r>=0; r--)
		{
			if (v(r) > z(Rx))
			{
				left = left+1;
				val_left = val_left+v(r);
			}
			else break;
		}
		
		FloatNDArray dum(dim_vector(2*delta+left+1, 1), val_left/(left+2*delta));
		v.insert(dum, floor(fmax((float)Rx-delta-left, 0)), 0);
	}
	
	return v;
}

//FloatNDArray projectSlopeQuadProg(FloatNDArray z, octave_int32 Rx)
//{
//	FloatNDArray v(z.dims());
//	int numZ = (int)z.dim1();
//	
//	QuadProgPP::Matrix<double> G((double)0, numZ, numZ);
//	QuadProgPP::Vector<double> g0(numZ);
//	QuadProgPP::Matrix<double> CE(numZ, 0);
//	QuadProgPP::Vector<double> ce0(0);
//	QuadProgPP::Matrix<double> CI((double)0, numZ, numZ-1);
//	QuadProgPP::Vector<double> ci0((double)0, numZ-1);
//	QuadProgPP::Vector<double> x(numZ);
//	
//	for (int r=0; r<numZ; r++)
//	{
//		G[r][r] = 1;
//		g0[r] = -2*z(r);
//		
//		
//		if (r < numZ-1)
//		{
//			CI[r][r] = -1;
//			CI[r+1][r] = 1;
//		}
//	}
//	
//	solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
//	
//	for (int r=0; r<numZ; r++)
//	{
//		v(r) = x[r]/2;
//	}
//	
//	return v;
//}

/* ----------------------------------------------------------------------------*
 * FloatNDArray projectSlope(FloatNDArray z, octave_int32 Rx, float delta)     *
 * ----------------------------------------------------------------------------*
 * Project the vector z onto v.
 *
 * @param z
 * @param Rx
 * @param delta
 *
 * @return v
 *
 */
 
FloatNDArray projectSlope(FloatNDArray z, octave_int32 Rx,  float delta)
{
	return projectSlopeMeanLeveling(z, Rx, delta);
}
