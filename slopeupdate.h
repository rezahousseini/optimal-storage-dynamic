// Dependencies
//#include "linprog.h"

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
		if (Rx(m)+1 == v.size1()) {
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
	vector<float> xc, vector<float> xd)
{
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
	int upper = floor(fmin((float)Rx+delta, v.size1()-1));
	
	for (int m=lower; m<=upper; m++) {
		z(m) = (1-(1-gama)*alpha)*v(m)+alpha*vhat;
	}
	
	return z;
}

vector<float> projectSlopeLeveling(vector<float> z, int Rx, float delta) {
	FloatNDArray v(z);
	int numZ = (int)z.dim1();
	
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
	int numZ = z.size1();
	int left = 0;
	int right = 0;
	float val_right = z(Rx)*2*delta;
	float val_left = z(Rx)*2*delta;
	
	int lower = floor(fmax((float)Rx-1-delta, 0));
	int upper = floor(fmin((float)Rx+1+delta, numZ));
	
	for (int r=upper; r<numZ; r++) {
		if (v(r) < z(Rx)) {
			right = right+1;
			val_right = val_right+v(r);
		}
		else break;
	}
	
	if (right > 0) {
		vector<float> dum(2*delta+right+1) = scalar_vector<float>(2*delta+right+1, val_right/(right+2*delta));
		v.insert(dum, floor(fmax((float)Rx-delta, 0)), 0);
	}
	else {
		for (int r=lower; r>=0; r--) {
			if (v(r) > z(Rx)) {
				left = left+1;
				val_left = val_left+v(r);
			}
			else break;
		}
		
		vector<float> dum(2*delta+left+1) = scalar_vector<float>(2*delta+left+1, val_left/(left+2*delta));
		
		v.insert(dum, floor(fmax((float)Rx-delta-left, 0)), 0);
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
