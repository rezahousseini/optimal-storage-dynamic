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
//		z(m) = (1-alpha)*v(Rx)+alpha*vhat;
	}
	
	return z;
}

FloatNDArray projectSlopeLeveling(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z);
	int numZ = (int)z.dim1();
	
	for (int r=(int)Rx+1; r<numZ; r++)
	{
		if (v(r) < z(Rx))
		{
			v(r) = z(Rx);
		}
	}
	
	for (int r=(int)Rx-1; r>=0; r--)
	{
		if (v(r) > z(Rx))
		{
			v(r) = z(Rx);
		}
	}
	
	return v;
}

FloatNDArray projectSlopeMeanLeveling(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z);
	int numZ = (int)z.dim1();
	int violation = 0;
	FloatNDArray sumVec;
	float sum;
	
	// r < Rx
	for (int level=(int)Rx-1; level>=0; level--)
	{
		if (z(level) < z(Rx))
		{
			violation = (int)Rx-level;
			sumVec = z.linear_slice(level, (int)Rx);
			sum = sumVec.sum(0).elem(0);
		}
	}
	
	if (violation > 0)
	{
		FloatNDArray vmean(dim_vector(violation+1, 1), sum/(violation+1));
		v.insert(vmean, (int)Rx-violation, 0);
	}
	
	violation = 0;
	// r > Rx
	for (int level=(int)Rx+1; level<numZ; level++)
	{
		if (z(level) > v(Rx))
		{
			violation = level-(int)Rx;
			sumVec = z.linear_slice(Rx, level);
			sum = sumVec.sum(0).elem(0);
		}
	}
	
	if (violation > 0)
	{
		FloatNDArray vmean;
		if (sum/(violation+1) > (float)v.min().elem(0))
		{
			vmean = FloatNDArray(dim_vector(numZ-(int)Rx, 1), (float)v.min().elem(0));
		}
		else
		{
			vmean = FloatNDArray(dim_vector(violation+1, 1), sum/(violation+1));
		}
		
		v.insert(vmean, Rx, 0);
	}
	
	return v;
}

FloatNDArray projectSlopeQuadProg(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z.dims());
	int numZ = (int)z.dim1();
	
	QuadProgPP::Matrix<double> G((double)0, numZ, numZ);
	QuadProgPP::Vector<double> g0(numZ);
	QuadProgPP::Matrix<double> CE(numZ, 0);
	QuadProgPP::Vector<double> ce0(0);
	QuadProgPP::Matrix<double> CI((double)0, numZ, numZ-1);
	QuadProgPP::Vector<double> ci0((double)0, numZ-1);
	QuadProgPP::Vector<double> x(numZ);
	
	for (int r=0; r<numZ; r++)
	{
		G[r][r] = 1;
		g0[r] = -2*z(r);
		
		
		if (r < numZ-1)
		{
			CI[r][r] = -1;
			CI[r+1][r] = 1;
		}
	}
	
	solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
	
	for (int r=0; r<numZ; r++)
	{
		v(r) = x[r]/2;
	}
	
	return v;
}

FloatNDArray simpleLinReg(FloatNDArray y, float x0, float x1)
{
	float xmean = (x1-x0)/2*(x0+x1)/(x1-x0);
	float ymean = y.sum(0).elem(0)/(float)y.dim1();
	FloatNDArray c(dim_vector(2, 1));
	
	float c1N = 0;
	float c1D = 0;
	
//	printf("y=");
//	for (int m=0; m<(int)y.dim1(); m++)
//	{
//		printf("%f ", y(m));
//	}
//	printf("\n");
	
	for (int x=0; x<(int)y.dim1(); x++)
	{
		c1N = c1N+((float)x-xmean)*(y(x)-ymean);
		c1D = c1D+pow((float)x-xmean, 2);
	}
	
	if (x0 == x1)
	{
		c(1) = 0;
		c(0) = 0;
	}
	else
	{
		c(1) = c1N/c1D;
		c(0) = ymean-c(1)*xmean;
	}
	
//	printf("c0=%f\n", c(0));
//	printf("c1=%f\n", c(1));
	
	return c;
}

FloatNDArray projectSlopeLinReg(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z.dims());
	int numZ = (int)z.dim1();
	
	FloatNDArray c;
	
	int step = 10;
	int n = floor(numZ/step);
	
	FloatNDArray C(dim_vector(2, step));
	
	for (int r=step-1; r>=0; r--)
	{
		if (r == step-1)
		{
			c = simpleLinReg(z.linear_slice(r*n, numZ), r*n, numZ-1);
			
			if (c(1) > 0) C(1, r) = 0;
			else C(1, r) = c(1);
			
			C(0, r) = c(0);
			
			for (int x=n*r; x<numZ; x++)
			{
				v(x) = C(0, r)+C(1, r)*(float)x;
			}
		}
		else
		{
			c = simpleLinReg(z.linear_slice(r*n, r*n+n), r*n, r*n+n);
			
			if (c(1) > 0) C(1, r) = C(1, r+1);
			else C(1, r) = c(1);
			
			C(0, r) = C(0, r+1)+C(1, r+1)*(r+1)*n-C(1, r)*(r+1)*n;
			
			for (int x=n*r; x<n*(r+1); x++)
			{
				v(x) = C(0, r)+C(1, r)*(float)x;
			}
		}
	}
	
	return v;
}

/* ----------------------------------------------------------------------------*
 * FloatNDArray projectSlope(FloatNDArray z, octave_int32 Rx)                  *
 * ----------------------------------------------------------------------------*
 * Project the vector z onto v.
 *
 * @param z
 * @param Rx
 *
 * @return v
 *
 */
 
FloatNDArray projectSlope(FloatNDArray z, octave_int32 Rx)
{
//	printf("Rx=%i\n", (int)Rx);
//	
//	printf("z=");
//	for (int m=0; m<(int)z.dim1(); m++)
//	{
//		printf("%f ", z(m));
//	}
//	printf("\n");
	
	FloatNDArray v = projectSlopeLeveling(z, Rx);
	
//	printf("v=");
//	for (int m=0; m<(int)z.dim1(); m++)
//	{
//		printf("%f ", v(m));
//	}
//	printf("\n");

	return v;
}
