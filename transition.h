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
			Rxerr = etal(m)*(float)R(count)+T*(etac(m)*xc(m)-xd(m)/etad(m));
			
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
	
	return Rx;
}
