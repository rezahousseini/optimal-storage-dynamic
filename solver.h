#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <glpk.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/numeric.hpp>

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
float T;
float rho;
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
//#include "transition.h"
//#include "slopeupdate.h"
#include "utils.h"

solution solve(float rho, matrix<float> g, matrix<float> r, prices P, storages S, int numI, float T, parameters parm) {
	// Parameters
	gama = parm.gama;
	alpha0 = parm.alpha0;
	deltaStepMult = parm.deltaStepMult;
	a = parm.a;
	b = parm.b;
	c = parm.c;
	float epsilon = parm.epsilon;
	
	// Prices
	matrix<float> pg = P.pg;
	matrix<float> pr = P.pr;
	vector<matrix<float> > pc = P.pc;
	vector<matrix<float> > pd = P.pd;
	
	numN = g.size1();
	numW = g.size2();
	
	// Pre-decision asset level
	matrix<int> R = init(S);
	
	// Post-decision asset level
	matrix<int> Rx(numSfin, numN);
	
	// Value function for the different levels
	int sumNumR = accumulate(numR, 0);
	matrix<float> v(sumNumR, numN);
	v = zero_matrix<float>(sumNumR, numN);
	
	opt_sol ret;
	vector<int> smpl;
	matrix<float> xc(numS, numN);
	xc = zero_matrix<float>(numS, numN);
	matrix<float> xd(numS, numN);
	xd = zero_matrix<float>(numS, numN);
	
	vector<float> alpha(numN);
	matrix<float> deltaStep(numSfin, numN);
	for (int m=0; m<numSfin; m++) {
		for (int k=0; k<numN; k++) {
			deltaStep(m, k) = deltaStepMult*(float)numR(m);
		}
	}
	
	// Return values
	solution sol;
	matrix<float> cost(numN, numI);
	cost = zero_matrix<float>(numN, numI);
	
	initLinProg();
	
	for (int i=1; i<numI; i++) {
		smpl = randi(0, numW, numN); // Generate sample
		
		for (int k=0; k<numN; k++) {
		matrix_column<matrix<float> > mc(P.pc(smpl(k)), k);
		std::cout << mc << std::endl;
			// Find optimal value function
//			ret = solveLinProg(
//				g(k, smpl(k)), r(k, smpl(k)),
//				P.pc.page(smpl(k)).column(k),
//				P.pd.page(smpl(k)).column(k),
//				matrix_column<matrix<float> > (R, k),
//				matrix_column<matrix<float> > (v, k),
//				matrix_column<matrix<float> > (xc, k),
//				matrix_column<matrix<float> > (xd, k)
//			);
//			
//			xc.insert(ret.xc, 0, k);
//			xd.insert(ret.xd, 0, k);
//			
//			// Resource transition function
//			Rx.insert(transitionResource(R.column(k), xc.column(k), xd.column(k)), 0, k);
//			
//			cost(k, i) = g(k, smpl(k))*pg(k, smpl(k))+r(k, smpl(k))*pr(k, smpl(k));
//			for (int m=0; m<numS; m++) {
//				cost(k, i) = cost(k, i)+xc(m, k)/rho*pc(m, k, smpl(k))+xd(m, k)/rho*pd(m, k, smpl(k));
//			}
//			
//			alpha(k) = alpha0*(b/i+a)/(b/i+a+pow(i, c));
//			
//			if (k < numN-1) {
//				// Compute post-decision asset level
//				matrix_column<matrix<float> > (R, k+1) = matrix_column<matrix<float> > (Rx, k);
//				
//				// Observe slope
//				vhat = observeSlope(g(k+1, smpl(k+1)), r(k+1, smpl(k+1)),
//					pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
//					Rx.column(k), v.column(k+1), xc.column(k+1), xd.column(k+1)
//				);
//				
//				
//				int index = 0;
//				for (int m=0; m<numSfin; m++) {
//					// Update slope
//					z = updateSlope(
//						v.column(k).linear_slice(index, index+numR(m)),
//						vhat(m), alpha(k), Rx(m, k), deltaStep(m, k)
//					);
//					
//					// Project slope
//					v.insert(projectSlope(z, Rx(m, k), deltaStep(m, k)), index, k);
//					
//					index = index+numR(m);
//				} // endfor m
//			} // endif
		} // endfor k
//		
//		for (int m=0; m<numSfin; m++) {
//			for (int k=0; k<numN; k++) {
//				if (cost(k, i) >= cost(k, i-1)+epsilon) {
//					deltaStep(m, k) = fmax(0, 0.5*deltaStep(m, k));
//				}
//			}
//		}
	} // endfor iter
	
	deleteLinProg();
	
//	// Rescale return value
//	sol.q = Rx/rho;
//	sol.uc = xc/rho;
//	sol.ud = xd/rho;
//	sol.cost = matrix_column<matrix<float> >(cost, numI-1);
//	sol.costIter = cost;
	
	return sol;
}
