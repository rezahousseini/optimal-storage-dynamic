/* ----------------------------------------------------------------------------*
 * int32NDArray transitionResource(int32NDArray R, FloatNDArray xc,            *
 *  FloatNDArray xd)                                                           *
 * ----------------------------------------------------------------------------*
 * Resource transition function.
 *
 * @param R
 * @param xc
 * @param xd
 *
 * @return Rx
 *
 */

int32NDArray transitionResource(int32NDArray R, FloatNDArray xc,
	FloatNDArray xd)
{
	int32NDArray Rx(dim_vector(numSfin, 1));
	float Rxerr;
	int count = 0;
	
	for (int m=0; m<numS; m++)
	{
		if ((int)set_fin(m) == 1)
		{
			// Resource transition function
//			Rxerr = nul(m)*(float)R(count)+T*(nuc(m)*xc(m)-(1/nud(m))*xd(m));
			Rxerr = (float)R(count)+T*(xc(m)-xd(m));
			
			if (Rxerr < rho*Qmin(m))
			{
				Rx(count) = floor(rho*Qmin(m));
			}
			else if (Rxerr > rho*Qmax(m))
			{
				Rx(count) = floor(rho*Qmax(m));
			}
			else Rx(count) = floor(Rxerr);
			
			count = count+1;
		}
	}
	
//	printf("Rx=");
//	for (int m=0; m<numSfin; m++)
//	{
//		 printf("%i ", (int)Rx(m));
//	}
//	printf("\n");
	
	return Rx;
}
