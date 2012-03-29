/* ----------------------------------------------------------------------------*
 * matrix<int> init(storages S)                                                *
 * ----------------------------------------------------------------------------*
 * Initiation of the algorithm.
 *
 * @param S Structure with storage parameters.
 *
 * @return R
 *
 */ 

matrix<int> init(storages S) {
	// Storage parameters
	Qmax = S.Qmax;
	Qmin = S.Qmin;
	q0 = S.q0;
	C = S.C;
	D = S.D;
	etal = S.etal;
	etac = S.etac;
	etad = S.etad;
	DeltaCmax = S.DeltaCmax;
	DeltaDmax = S.DeltaDmax;
	
	// Number of resources
	numS = Qmax.size();
	set_fin = vector<int> (numS);
//	set_fin = zero_vector<int> (numS);
	
	int count = 0;
	for (int k=0; k<numS; k++) {
		// Qmax < inf
		if (Qmax(k) != 1.0/0.0) {
			set_fin(k) = 1;
		}
		else {
			set_fin(k) = 0;
		}
	}
	
	numSfin = accumulate(set_fin, 0);
	numR = vector<int> (numSfin);
	
	matrix<int> R(numSfin, numN);
//	R = zero_matrix<int> (numSfin, numN);
	
	count = 0;
	for (int k=0; k<numS; k++) {
		if (set_fin(k) == 1) {
			numR(count) = floor(rho*Qmax(k))+1; // Scale max capacity
			R(count, 0) = floor(rho*q0(k)); // Storage level initialization; TODO checking for q0 <= Qmax
			count = count+1;
		}
	}
	
//	srand(time(NULL));
	
	return R;
}
