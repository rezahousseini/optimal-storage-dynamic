vector<float> observeSlopeDual(float g, float r,
	vector<float> pc, vector<float> pd, 
	vector<int> Rx, vector<float> v,
	vector<float> xc, vector<float> xd) {
	
	opt_sol retce = solveLinProg(g, r, pc, pd, Rx, v, xc, xd);
	
	return retce.vhat;
}

vector<float> observeSlopeDerivative(float g, float r,
	vector<float> pc, vector<float> pd, 
	vector<int> Rx, vector<float> v,
	vector<float> xc, vector<float> xd) {
	
	vector<float> vhat(numSfin);
	opt_sol retup, retlo, retce;
	vector<int> Rxlo(Rx);
	vector<int> Rxup(Rx);
	
	retce = solveLinProg(g, r, pc, pd, Rx, v, xc, xd);
	
	for (int m=0; m<numSfin; m++) {
		if (Rx(m)+1 == v.size()) {
			// There's no value in adding more than the maximum
			vhat(m) = 0;
		}
		else {
			Rxup(m) = Rx(m)+1;
			
			retup = solveLinProg(g, r, pc, pd, Rxup, v, xc, xd);
			
			vhat(m) = retup.F-retce.F;
		}
	}
	
	return vhat;
}

/* ----------------------------------------------------------------------------*
 * vector<float> observeSlope(float g, float r, vector<float> pc,                *
 *  vector<float> pd, vector<int> Rx, vector<float> v,                          *
 *  vector<float> xc, vector<float> xd)                                          *
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

vector<float> observeSlope(float g, float r,
	vector<float> pc, vector<float> pd, 
	vector<int> Rx, vector<float> v,
	vector<float> xc, vector<float> xd) {
	return observeSlopeDual(g, r, pc, pd, Rx, v, xc, xd);
}

/* ----------------------------------------------------------------------------*
 * vector<float> updateSlope(vector<float> v, float vhat, float alpha, int Rx) *
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

vector<float> updateSlope(vector<float> v, float vhat, float alpha, int Rx,
	float delta) {
	vector<float> z(v);
	
	int lower = floor(fmax((float)Rx-delta, 0));
	int upper = floor(fmin((float)Rx+delta, v.size()-1));
	
	for (int m=lower; m<=upper; m++) {
		z(m) = (1-(1-parm.gama)*alpha)*v(m)+alpha*vhat;
	}
	
	return z;
}

vector<float> projectSlopeLeveling(vector<float> z, int Rx, float delta) {
	vector<float> v(z);
	int numZ = z.size();
	
	int lower = floor(fmax((float)Rx-1-delta, 0));
	int upper = floor(fmin((float)Rx+1+delta, numZ));
	
	for (int r=upper; r<numZ; r++) {
		if (v(r) < z(Rx)) {
			v(r) = z(Rx);
		}
		else break;
	}
	
	for (int r=lower; r>=0; r--) {
		if (v(r) > z(Rx)) {
			v(r) = z(Rx);
		}
		else break;
	}
	
	return v;
}

vector<float> projectSlopeMeanLeveling(vector<float> z, int Rx, float delta) {
	vector<float> v(z);
	int numZ = z.size();
	int left = 0;
	int right = 0;
	int lower = fmax(Rx-1-floor(delta), 0);
	int upper = fmin(Rx+1+floor(delta), numZ-1);
	float val_right = z(Rx)*2*(upper-Rx);
	float val_left = z(Rx)*2*(Rx-lower);
	
	for (int r=upper; r<numZ; r++) {
		if (v(r) < z(Rx)) {
			right = right+1;
			val_right = val_right+v(r);
		}
		else break;
	}
	
	if (right > 0) {
		vector<float> dum = scalar_vector<float> (upper-lower+right+1, val_right/(right+upper-lower));
		vector_slice<vector<float> > (v, slice(lower, 1, upper-lower+right+1)) = dum;
	}
	else {
		for (int r=lower; r>=0; r--) {
			if (v(r) > z(Rx)) {
				left = left+1;
				val_left = val_left+v(r);
			}
			else break;
		}
		
		vector<float> dum = scalar_vector<float> (upper-lower+left+1, val_left/(left+upper-lower));
		vector_slice<vector<float> > (v, slice(lower-left, 1, upper-lower+left+1)) = dum;
	}
	
	return v;
}

/* ----------------------------------------------------------------------------*
 * vector<float> projectSlope(vector<float> z, int Rx, float delta)            *
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
 
vector<float> projectSlope(vector<float> z, int Rx,  float delta) {
	return projectSlopeMeanLeveling(z, Rx, delta);
}

/* ----------------------------------------------------------------------------*
 * vector<float> update(float g, float r, vector<float> pc, vector<float> pd,  *
 *  vector<int> Rx, vector<float> v_old, vector<float> v_new,                  *
 *  vector<float> xc, vector<float> xd, vector<float> deltaStep,               *
 *  float alpha)                                                               *
 * ----------------------------------------------------------------------------*
 * Update v.
 *
 * @param z
 * @param Rx
 * @param delta
 *
 * @return v
 *
 */
 
vector<float> update(float g, float r, vector<float> pc, vector<float> pd,
	vector<int> Rx, vector<float> v_old, vector<float> v_new,
	vector<float> xc, vector<float> xd, vector<float> deltaStep, float alpha) {
	vector<float> z;
	vector<float> v(accumulate(numR, 0));
	
	// Observe slope
	vector<float> vhat = observeSlope(g, r, pc, pd,  Rx, v_new, xc, xd);
	
	int index = 0;
	for (int m=0; m<numSfin; m++) {
		// Update slope
		z = updateSlope(
			vector_slice<vector<float> > (v_old, slice(index, 1, index+numR(m))),
			vhat(m), alpha, Rx(m), deltaStep(m)
		);
		
		// Project slope
		vector_slice<vector<float> > (v, slice(index, 1, index+numR(m))) = 
			projectSlope(z, Rx(m), deltaStep(m));
		
		index = index+numR(m);
	} // endfor m
	
	return v;
}
