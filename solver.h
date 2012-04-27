#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <numeric>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/numeric.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <glpk.h>

using namespace boost;
using namespace boost::math;
using namespace boost::numeric::ublas;

// Storage parameters
struct storages {
	vector<float> Qmax;
	vector<float> Qmin;
	vector<float> q0;
	vector<float> C;
	vector<float> D;
	vector<float> etal;
	vector<float> etac;
	vector<float> etad;
	vector<float> DeltaCmax;
	vector<float> DeltaDmax;
};

// Prices
struct prices {
	matrix<float> pg;
	matrix<float> pr;
	vector<matrix<float> > pc;
	vector<matrix<float> > pd;
};

// Solver parameters
struct parameters {
	float gama;
	float alpha0;
	float deltaStepMult;
	float a;
	float b;
	float c;
	float epsilon;
};

// Return values
struct solution {
	matrix<float> q;
	matrix<float> uc;
	matrix<float> ud;
	vector<float> cost;
	matrix<float> costIter;
};

float rho;
storages S;
float T;
parameters parm;

glp_prob *lp;
glp_smcp parm_lp;
vector<int> numR;
int numS;
int numSfin;
int numW;
int numN;
vector<int> set_fin;

// Own source files.
#include "init.h"
#include "linprog.h"
#include "transition.h"
#include "slopeupdate.h"
#include "utils.h"
//#include "OSAVIStepsize.h"
#include "STCStepsize.h"

solution solve(float _rho, matrix<float> _g, matrix<float> _r, prices _P, storages _S, int _numI, float _T, parameters _parm) {
	rho = _rho;
	S = _S;
	T = _T;
	parm = _parm;
	numN = _g.size1();
	numW = _g.size2();
	
	// Pre-decision asset level
	matrix<int> R = init();
	
	// Value function for the different levels
	matrix<float> v = zero_matrix<float> (accumulate(numR, 0), numN);
	
	opt_sol ret;
	vector<int> smpl;
	matrix<float> xc = zero_matrix<float>(numS, numN);
	matrix<float> xd = zero_matrix<float>(numS, numN);
	matrix<float> cost = zero_matrix<float>(numN, _numI);
	
	STCStepsize stepsize(parm.alpha0, parm.c, parm.a, parm.b);
	//OSAVIStepsize stepsize(1, 1, 0.2, 1, parm.gama);
	matrix<float> deltaStep(numSfin, numN);
	for (int t=0; t<numN; t++) {
		matrix_column<matrix<float> > (deltaStep, t) = parm.deltaStepMult*numR;
	}
	
	// Return values
	solution sol;
	
	initLinProg();
	
	// Iteration loop
	for (int i=1; i<_numI; i++) {
		smpl = randi(0, numW, numN); // Generate sample
		
		// Time loop
		for (int t=0; t<numN; t++) {
			// Find optimal decisions
			ret = solveLinProg(
				_g(t, smpl(t)), _r(t, smpl(t)),
				matrix_column<matrix<float> > (_P.pc(smpl(t)), t),
				matrix_column<matrix<float> > (_P.pd(smpl(t)), t),
				matrix_column<matrix<int> > (R, t),
				matrix_column<matrix<float> > (v, t),
				matrix_column<matrix<float> > (xc, t),
				matrix_column<matrix<float> > (xd, t)
			);
			
			// Update decisions
			matrix_column<matrix<float> > (xc, t) = ret.xc;
			matrix_column<matrix<float> > (xd, t) = ret.xd;
			
			if (t < numN-1) {
				// Resource transition function
				matrix_column<matrix<int> > (R, t+1) = transitionResource(
					matrix_column<matrix<int> > (R, t),
					matrix_column<matrix<float> > (xc, t),
					matrix_column<matrix<float> > (xd, t)
				);
				
				// Update cost function
				matrix_column<matrix<float> > (v, t) = update(
					_g(t+1, smpl(t+1)), _r(t+1, smpl(t+1)),
					matrix_column<matrix<float> > (_P.pc(smpl(t+1)), t+1),
					matrix_column<matrix<float> > (_P.pd(smpl(t+1)), t+1),
					matrix_column<matrix<int> > (R, t+1),
					matrix_column<matrix<float> > (v, t),
					matrix_column<matrix<float> > (v, t+1),
					matrix_column<matrix<float> > (xc, t+1),
					matrix_column<matrix<float> > (xd, t+1),
					matrix_column<matrix<float> > (deltaStep, t),
					stepsize.get()
				);
			} // endif
			
			// Update cost and deltaStep
			cost(t, i) = _g(t, smpl(t))*_P.pg(t, smpl(t))+_r(t, smpl(t))*_P.pr(t, smpl(t));
			for (int m=0; m<numS; m++) {
				cost(t, i) = cost(t, i)+xc(m, t)/rho*_P.pc(smpl(t))(m, t)+xd(m, t)/rho*_P.pd(smpl(t))(m, t);
			}
			
			for (int m=0; m<numSfin; m++) {
				if (cost(t, i) >= cost(t, i-1)+parm.epsilon) {
					deltaStep(m, t) = fmax(0, 0.5*deltaStep(m, t));
				}
			}
		} // endfor t
		
		// Update stepsize
		//stepsize.update(accumulate(matrix_column<matrix<float> > (cost, i), 0));
		stepsize.update();
	} // endfor i
	
	// Free memory
	deleteLinProg();
	
	// Rescale return value
	sol.q = static_cast<matrix<float> >(R)/rho;
	sol.uc = xc/rho;
	sol.ud = xd/rho;
	sol.cost = matrix_column<matrix<float> > (cost, _numI-1);
	sol.costIter = cost;
	
	return sol;
}
