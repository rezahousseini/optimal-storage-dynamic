/* ----------------------------------------------------------------------------*
 * FloatNDArray observeslope(FloatNDArray z)                                        *
 * ----------------------------------------------------------------------------*
 * Observe the slopes.
 *
 * @param z
 *
 * @return v
 *
 */

FloatNDArray observeslope(float g, float r,
	FloatNDArray pc, FloatNDArray pd, 
	int32NDArray Rx, FloatNDArray v,
	int32NDArray xc, int32NDArray xd)
{
	FloatNDArray vhat(dim_vector(2, numSfin));
	opt_sol retup, retlo, retce;
	int32NDArray Rxlo(dim_vector(numSfin,1));
	int32NDArray Rxup(dim_vector(numSfin,1));
	
	retce = solveOpt(g, r, pc, pd, Rx, v, xc, xd);
	
	for (int m=0; m<numSfin; m++)
	{
		if (Rx(m) == (octave_int32)0)
		{
			Rxup.insert(Rx, 0, 0);
			Rxup(m) = Rx(m)+(octave_int32)1;
			
			retup = solveOpt(g, r, pc, pd, Rxup, v, xc, xd);
			
			vhat(0, m) = (octave_int32)0;//retce.V;
			vhat(1, m) = retup.F-retce.F;
		}
		else if (Rx(m) == floor(rho*Qmax(m)))
		{
			Rxlo.insert(Rx, 0, 0);
			Rxlo(m) = Rx(m)-(octave_int32)1;
			
			retlo = solveOpt(g, r, pc, pd, Rxlo, v, xc, xd);
			
			vhat(0, m) = retce.F-retlo.F;
			vhat(1, m) = -retce.F;
		}
		else
		{
			Rxlo.insert(Rx, 0, 0);
			Rxup.insert(Rx, 0, 0);
			Rxlo(m) = Rx(m)-(octave_int32)1;
			Rxup(m) = Rx(m)+(octave_int32)1;
			
			retlo = solveOpt(g, r, pc, pd, Rxlo, v, xc, xd);
			retup = solveOpt(g, r, pc, pd, Rxup, v, xc, xd);
			
			vhat(0, m) = retce.F-retlo.F;
			vhat(1, m) = retup.F-retce.F;
		}
		
		return vhat;
	}
}

/* ----------------------------------------------------------------------------*
 * FloatNDArray updateslope(FloatNDArray z)                                        *
 * ----------------------------------------------------------------------------*
 * Update the vector z with v.
 *
 * @param z
 *
 * @return z
 *
 */

FloatNDArray updateslope(FloatNDArray v, FloatNDArray vhat, float alpha, octave_int32 Rx)
{
	FloatNDArray z(v.dims());
	
	z.insert(v, 0, 0);
	
	z(Rx) = (1-(1-gama)*alpha)*v(Rx)+alpha*vhat(0);
	
	if (Rx+(octave_int32)1 < (octave_int32)v.dim1())
	{
		z(Rx+(octave_int32)1) = (1-(1-gama)*alpha)*v(Rx+(octave_int32)1)+alpha*vhat(1);
	}
	
	return z;
}

/* ----------------------------------------------------------------------------*
 * FloatNDArray projectslope(FloatNDArray z)                                        *
 * ----------------------------------------------------------------------------*
 * Project the vector z onto v.
 *
 * @param z
 *
 * @return v
 *
 */
 
FloatNDArray projectslope(FloatNDArray z, octave_int32 Rx)
{
	FloatNDArray v(z.dims());
	int numZ = (int)z.dim1();
	
//	printf("z=");
//	for (int r=0; r<numZ; r++)
//	{
//		printf("%f ", z(r));
//	}
//	printf("\n");
	
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
	
//	CE[(int)Rx][0] = 1;
//	ce0[0] = -z(Rx);
	
	solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
	
	for (int r=0; r<numZ; r++)
	{
		v(r) = x[r];
	}
	
//	if (v(index+Rx(m, k), k) < v(index+Rx(m, k)+(octave_int32)1, k))
//	{
//		v(index+Rx(m, k), k) = v(index+Rx(m, k)+(octave_int32)1, k);
//	}
	
//	// r > Rx
////	if (Rx(m, k)+(octave_int32)1 < numR(m))
////	{
//		for (int level=(int)Rx(m, k)+1; level < (int)numR(m)-1; level++)
//		{
//			if (v((int)index+level+1, k) <= v((int)index+level, k))
//			{
//				break;
//			}
//			else
//			{
//				float vmean = (v((int)index+level+1, k)+(level-(int)Rx(m, k))*v((int)index+level, k))/(level-(int)Rx(m, k)+1);
//				for (int j=(int)Rx(m, k)+1; j<=level+1; j++)
//				{
//					v((int)index+j, k) = vmean;//v(index+Rx(m, k)+(octave_int32)1, k);
//				}
//			}
//		}
//	}
//	
//	if (v(index+Rx(m, k), k) < v(index+Rx(m, k)+(octave_int32)1, k))
//	{
//		v(index+Rx(m, k), k) = v(index+Rx(m, k)+(octave_int32)1, k);
//	}
//	
//	// Projection operation
//	// r < Rx
//	for (int level=(int)Rx(m, k); level>0; level--)
//	{
//		if(v((int)index+level-1, k) >= v((int)index+level, k))
//		{
//			break;
//		}
//		else
//		{
//			float vmean = (v((int)index+level-1, k)+((int)Rx(m, k)-level+1)*v((int)index+level, k))/((int)Rx(m, k)-level+2);
//			for (int j=(int)Rx(m, k); j>=level-1; j--)
//			{
//				v((int)index+j, k) = vmean;//v(index+Rx(m, k), k);
//			}
//		}
//	}
	
//	v.insert(z, 0, 0);
//	
//	for (int r=0; r<numZ; r++)
//	{
//		if ((v(r) < z(Rx) and r < (int)Rx) or (v(r) > z(Rx) and r > (int)Rx))
//		{
//			
//			v(r) = z(Rx);
//		}
//	}
	
//	printf("v=");
//	for (int r=0; r<numZ; r++)
//	{
//		printf("%f ", v(r));
//	}
//	printf("\n");
	
	return v;
}
