#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <glpk.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/numeric.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

using namespace boost;
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

vector<int> numR;
int numS;
int numSfin;
int numW;
int numN;
glp_prob *lp;
glp_smcp parm_lp;
float rho;
storages S;
float T;
parameters parm;
vector<int> set_fin;

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

float gama = 0.5; // 0 <= gamma <= 1 0.5
float alpha0 = 0.8; // 0 <= alpha0 <= 1 0.8
float deltaStepMult = 0.8; // 0.8
float a = 4; // 4
float b = 100; // 40
float c = 0.5; // 0.4

// Own source files.
#include "init.h"
#include "linprog.h"
#include "transition.h"
#include "slopeupdate.h"
#include "utils.h"

solution solve(float rho, matrix<float> g, matrix<float> r, prices P, storages S, int numI, float T, parameters parm) {
	numN = g.size1();
	numW = g.size2();
	
	// Pre-decision asset level
	matrix<int> R = init();
	
	// Value function for the different levels
	matrix<float> v = zero_matrix<float> (accumulate(numR, 0), numN);
	
	opt_sol ret;
	vector<int> smpl;
	matrix<float> xc = zero_matrix<float>(numS, numN);
	matrix<float> xd = zero_matrix<float>(numS, numN);
	matrix<float> cost = zero_matrix<float>(numN, numI);
	
	vector<float> alpha(numN);
	matrix<float> deltaStep(numSfin, numN);
	for (int m=0; m<numSfin; m++) {
		for (int k=0; k<numN; k++) {
			deltaStep(m, k) = parm.deltaStepMult*(float)numR(m);
		}
	}
	
	// Return values
	solution sol;
	
	initLinProg();
	
	for (int i=1; i<numI; i++) {
		smpl = randi(0, numW, numN); // Generate sample
		
		for (int k=0; k<numN; k++) {
			// Find optimal value function
			ret = solveLinProg(
				g(k, smpl(k)), r(k, smpl(k)),
				matrix_column<matrix<float> > (P.pc(smpl(k)), k),
				matrix_column<matrix<float> > (P.pd(smpl(k)), k),
				matrix_column<matrix<int> > (R, k),
				matrix_column<matrix<float> > (v, k),
				matrix_column<matrix<float> > (xc, k),
				matrix_column<matrix<float> > (xd, k)
			);
			
			matrix_column<matrix<float> > (xc, k) = ret.xc;
			matrix_column<matrix<float> > (xd, k) = ret.xd;
			
			cost(k, i) = g(k, smpl(k))*P.pg(k, smpl(k))+r(k, smpl(k))*P.pr(k, smpl(k));
			for (int m=0; m<numS; m++) {
				cost(k, i) = cost(k, i)+xc(m, k)/rho*P.pc(smpl(k))(m, k)+xd(m, k)/rho*P.pd(smpl(k))(m, k);
			}
			
			// Update stepsize
			alpha(k) = parm.alpha0*(parm.b/i+parm.a)/(parm.b/i+parm.a+pow(i, parm.c));
			
			if (k < numN-1) {
				// Resource transition function
				matrix_column<matrix<int> > (R, k+1) = transitionResource(
					matrix_column<matrix<int> > (R, k),
					matrix_column<matrix<float> > (xc, k),
					matrix_column<matrix<float> > (xd, k)
				);
				
				// Update cost function
				matrix_column<matrix<float> > (v, k) = update(
					g(k+1, smpl(k+1)), r(k+1, smpl(k+1)),
					matrix_column<matrix<float> > (P.pc(smpl(k+1)), k+1),
					matrix_column<matrix<float> > (P.pd(smpl(k+1)), k+1),
					matrix_column<matrix<int> > (R, k+1),
					matrix_column<matrix<float> > (v, k),
					matrix_column<matrix<float> > (v, k+1),
					matrix_column<matrix<float> > (xc, k+1),
					matrix_column<matrix<float> > (xd, k+1),
					matrix_column<matrix<float> > (deltaStep, k),
					alpha(k)
				);
			} // endif
		} // endfor k
		
		for (int m=0; m<numSfin; m++) {
			for (int k=0; k<numN; k++) {
				if (cost(k, i) >= cost(k, i-1)+parm.epsilon) {
					deltaStep(m, k) = fmax(0, 0.5*deltaStep(m, k));
				}
			}
		}
	} // endfor iter
	
	deleteLinProg();
	
	// Rescale return value
	sol.q = static_cast<matrix<float> >(R)/rho;
	sol.uc = xc/rho;
	sol.ud = xd/rho;
	sol.cost = matrix_column<matrix<float> > (cost, numI-1);
	sol.costIter = cost;
	
	return sol;
}
