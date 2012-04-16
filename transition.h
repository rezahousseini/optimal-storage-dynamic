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

vector<int> transitionResource(vector<int> R, vector<float> xc, vector<float> xd) {
	vector<int> Rx(numSfin);
	float Rxerr;
	int count = 0;
	
	for (int m=0; m<numS; m++) {
		if (set_fin(m) == 1) {
			// Resource transition function
			Rxerr = S.etal(m)*static_cast<float>(R(count))+T*(S.etac(m)*xc(m)-xd(m)/S.etad(m));
			
			if (Rxerr < rho*S.Qmin(m)) {
				Rx(count) = floor(rho*S.Qmin(m));
			}
			else if (Rxerr > rho*S.Qmax(m)) {
				Rx(count) = floor(rho*S.Qmax(m));
			}
			else Rx(count) = floor(Rxerr);
			
			count = count+1;
		}
	}
	
	return Rx;
}
