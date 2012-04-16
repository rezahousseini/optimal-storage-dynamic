/* ----------------------------------------------------------------------------*
 * matrix<int> init()                                                          *
 * ----------------------------------------------------------------------------*
 * Initiation of the algorithm.
 *
 * @param void
 *
 * @return R
 *
 */ 

matrix<int> init() {
	// Number of resources
	numS = S.Qmax.size();
	set_fin = vector<int> (numS);
//	set_fin = zero_vector<int> (numS);
	
	for (int k=0; k<numS; k++) {
		// Qmax < inf
		if (!isinf(S.Qmax(k))) set_fin(k) = 1;
	}
	
	numSfin = accumulate(set_fin, 0);
	numR = vector<int> (numSfin);
	
	matrix<int> R(numSfin, numN);
//	R = zero_matrix<int> (numSfin, numN);
	
	int count = 0;
	for (int k=0; k<numS; k++) {
		if (set_fin(k) == 1) {
			numR(count) = floor(rho*S.Qmax(k))+1; // Scale max capacity
			R(count, 0) = floor(rho*S.q0(k)); // Storage level initialization; TODO checking for q0 <= Qmax
			count = count+1;
		}
	}
	
	return R;
}
