/* ----------------------------------------------------------------------------*
 * FloatNDArray observeSlope(FloatNDArray z)                                   *
 * ----------------------------------------------------------------------------*
 * Observe the slopes.
 *
 * @param z
 *
 * @return v
 *
 */

FloatNDArray observeSlope(float g, float r,
	FloatNDArray pc, FloatNDArray pd, 
	int32NDArray Rx, FloatNDArray v,
	int32NDArray xc, int32NDArray xd)
{
	FloatNDArray vhat(dim_vector(1, numSfin));
	opt_sol retup, retlo, retce;
	int32NDArray Rxlo(dim_vector(numSfin,1));
	int32NDArray Rxup(dim_vector(numSfin,1));
	
	retce = solveLinProg(g, r, pc, pd, Rx, v, xc, xd);
	
	for (int m=0; m<numSfin; m++)
	{
//		if (Rx(m) == (octave_int32)0)
//		{
//			Rxup.insert(Rx, 0, 0);
//			Rxup(m) = Rx(m)+(octave_int32)1;
//			
//			retup = solveLinProg(g, r, pc, pd, Rxup, v, xc, xd);
//			
//			vhat(0, m) = (octave_int32)0;//retce.V;
//			vhat(1, m) = retup.F-retce.F;
//		}
//		else if (Rx(m) == floor(rho*Qmax(m)))
//		{
//			Rxlo.insert(Rx, 0, 0);
//			Rxlo(m) = Rx(m)-(octave_int32)1;
//			
//			retlo = solveLinProg(g, r, pc, pd, Rxlo, v, xc, xd);
//			
//			vhat(0, m) = retce.F-retlo.F;
//			vhat(1, m) = -retce.F;
//		}
//		else
//		{
//			Rxlo.insert(Rx, 0, 0);
//			Rxup.insert(Rx, 0, 0);
//			Rxlo(m) = Rx(m)-(octave_int32)1;
//			Rxup(m) = Rx(m)+(octave_int32)1;
//			
//			retlo = solveLinProg(g, r, pc, pd, Rxlo, v, xc, xd);
//			retup = solveLinProg(g, r, pc, pd, Rxup, v, xc, xd);
//			
//			vhat(0, m) = retce.F-retlo.F;
//			vhat(1, m) = retup.F-retce.F;
//		}
		
		if (Rx(m) == floor(rho*Qmax(m)))
		{
			vhat(0, m) = (octave_int32)0;
		}
		else
		{
			Rxup.insert(Rx, 0, 0);
			Rxup(m) = Rx(m)+(octave_int32)1;
			
			retup = solveLinProg(g, r, pc, pd, Rxup, v, xc, xd);
			
			vhat(0, m) = retup.F-retce.F;
		}
	}
	
	
	
//	printf("vhat=");
//	for (int h=0; h<1; h++)
//	{
//		 printf("%f ", vhat(h));
//	}
//	printf("\n");
	
	return vhat;
}

/* ----------------------------------------------------------------------------*
 * FloatNDArray updateSlope(FloatNDArray v, FloatNDArray vhat, float alpha,    *
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

FloatNDArray updateSlope(FloatNDArray v, FloatNDArray vhat, float alpha,
	octave_int32 Rx, float delta)
{
	FloatNDArray z(v.dims());
	
//	printf("delta=%f\n", delta);
	
	z.insert(v, 0, 0);
	
	for (int m=floor(fmax((float)Rx-delta, 0)); m<floor(fmin((float)Rx+delta, (float)v.dim1())); m++)
	{
		z(m) = (1-(1-gama)*alpha)*v(Rx)+alpha*vhat(0);
	}
	
	return z;
}

FloatNDArray projectSlopeLeveling(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z.dims());
	int numZ = (int)z.dim1();
	
	v.insert(z, 0, 0);
	
	for (int r=(int)Rx+1; r<numZ; r++)
	{
		if (v(r) > z(Rx))
		{
			
			v(r) = z(Rx);
		}
	}
	
	for (int r=(int)Rx-1; r>=0; r--)
	{
		if (v(r) < z(Rx))
		{
			
			v(r) = z(Rx);
		}
	}
	
	return v;
}

FloatNDArray projectSlopeMeanLeveling(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z.dims());
	int numZ = (int)z.dim1();
	int violation = 0;
	FloatNDArray sumVec;
	float sum;
	
//	printf("Rx=%i\n", (int)Rx);
//	
//	printf("z=");
//	for (int m=0; m<numZ; m++)
//	{
//		printf("%f ", z(m));
//	}
//	printf("\n");
	
	v.insert(z, 0, 0);
	
	// r < Rx
	for (int level=(int)Rx-1; level>=0; level--)
	{
		if (z(level) < z(Rx))
		{
			violation = (int)Rx-level;
			sumVec = z.linear_slice(level, (int)Rx+1);
			sum = sumVec.sum(0).elem(0);
		}
	}
	
	if (violation > 0)
	{
		FloatNDArray vmean(dim_vector(violation+1, 1), sum/(violation+1));
		v.insert(vmean, (int)Rx-violation, 0);
	}
	else
	{
		// r > Rx
		for (int level=(int)Rx+1; level<numZ; level++)
		{
			if (z(level) > z(Rx))
			{
				violation = level-(int)Rx;
				sumVec = z.linear_slice(Rx, level+1);
				sum = sumVec.sum(0).elem(0);
			}
		}
		
		if (violation > 0)
		{
			FloatNDArray vmean(dim_vector(violation+1, 1), sum/(violation+1));
			v.insert(vmean, Rx, 0);
		}
	}
	
//	printf("v=");
//	for (int m=0; m<numZ; m++)
//	{
//		printf("%f ", v(m));
//	}
//	printf("\n");
	
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
			CI[r][r] = 1;
			CI[r+1][r] = -1;
		}
	}
	
	solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
	
	for (int r=0; r<numZ; r++)
	{
		v(r) = x[r];
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
	return projectSlopeMeanLeveling(z, Rx);
}
