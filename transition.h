/* ----------------------------------------------------------------------------*
 * vector<int> transitionResource(vector<int> R, vector<float> xc,             *
 *  vector<float> xd)                                                          *
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

vector<int> transitionResource(vector<int> R, vector<float> xc,
	vector<float> xd) {
	vector<int> Rx(numSfin);
	float Rxerr;
	int count = 0;
	
	for (int m=0; m<numS; m++)
	{
		if (set_fin(m) == 1)
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
