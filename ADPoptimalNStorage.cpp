#include "mex.h"
#include "matrix.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "solver.h"
#include <iostream>

using namespace boost::numeric::ublas;

vector<float> mex2ublas1d(const mxArray *a_mex);
matrix<float> mex2ublas2d(const mxArray *a_mex);
vector<matrix<float> > mex2ublas3d(const mxArray *a_mex);
mxArray * ublas2mex1d(vector<float> a);
mxArray * ublas2mex2d(matrix<float> a);

/* ----------------------------------------------------------------------------*
 * void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])*
 * ----------------------------------------------------------------------------*
 * Main function. Call "ADPoptimalNStorage(rho, g, r, P, S, numI, T, parm)" in 
 * Matlab.
 *
 * @param rho Scaling factor for the quantisation.
 * @param g numW samples of generated energy for every time step numN.
 * @param r numW samples of requested energy for every time step numN.
 * @param P Structure with numW samples of energy cost for every time step numN.
 * @param S Structure with storage parameters.
 * @param numI Number of Iterations.
 * @param T Time step size.
 *
 * @return q, uc, ud, cost
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 8) mexErrMsgTxt("rho, g, r, Prices, Storages, numI, T, Parameters");
	else {
		double rho = mxGetScalar(prhs[0]);
		const mxArray *prod = prhs[1];
		const mxArray *cons = prhs[2];
		const mxArray *Prices = prhs[3];
		const mxArray *Storages = prhs[4];
		int numI = (int)mxGetScalar(prhs[5]);
		double T = mxGetScalar(prhs[6]);
		const mxArray *Parameters = prhs[7];
		
		parameters parm;
        prices P;
        storages S;
		
		// Parameters
		parm.gama = mxGetScalar(mxGetField(Parameters, 0, "gamma"));
		parm.alpha0 = mxGetScalar(mxGetField(Parameters, 0, "alpha0"));
		parm.deltaStepMult = mxGetScalar(mxGetField(Parameters, 0, "deltaStepMult"));
		parm.a = mxGetScalar(mxGetField(Parameters, 0, "a"));
		parm.b = mxGetScalar(mxGetField(Parameters, 0, "b"));
		parm.c = mxGetScalar(mxGetField(Parameters, 0, "c"));
		parm.epsilon = mxGetScalar(mxGetField(Parameters, 0, "epsilon"));
		
		matrix<float> g = mex2ublas2d(prod);
		matrix<float> r = mex2ublas2d(cons);
		
		// Prices
		P.pg = mex2ublas2d(mxGetField(Prices, 0, "pg"));
		P.pr = mex2ublas2d(mxGetField(Prices, 0, "pr"));
		P.pc = mex2ublas3d(mxGetField(Prices, 0, "pc"));
		P.pd = mex2ublas3d(mxGetField(Prices, 0, "pd"));
		
		S.Qmax = mex2ublas1d(mxGetField(Storages, 0, "Qmax"));
		S.Qmin = mex2ublas1d(mxGetField(Storages, 0, "Qmin"));
		S.q0 = mex2ublas1d(mxGetField(Storages, 0, "q0"));
		S.C = mex2ublas1d(mxGetField(Storages, 0, "C"));
		S.D = mex2ublas1d(mxGetField(Storages, 0, "D"));
		S.etal = mex2ublas1d(mxGetField(Storages, 0, "etal"));
		S.etac = mex2ublas1d(mxGetField(Storages, 0, "etac"));
		S.etad = mex2ublas1d(mxGetField(Storages, 0, "etad"));
		S.DeltaCmax = mex2ublas1d(mxGetField(Storages, 0, "DeltaCmax"));
		S.DeltaDmax = mex2ublas1d(mxGetField(Storages, 0, "DeltaDmax"));
		
		solution sol = solve(rho, g, r, P, S, numI, T, parm);
		
		// Return values
		plhs[0] = ublas2mex2d(sol.q);
		plhs[1] = ublas2mex2d(sol.uc);
		plhs[2] = ublas2mex2d(sol.ud);
		plhs[3] = ublas2mex1d(sol.cost);
		plhs[4] = ublas2mex2d(sol.costIter);
	}
	
	return;
}

vector<float> mex2ublas1d(const mxArray *a_mex) {
	const mwSize *d = mxGetDimensions(a_mex);
    int dim0 = d[0];
    int dim1 = d[1];
    int dim;
    
    if (dim0 > dim1) dim = dim0;
    else dim = dim1;
    
    double *a_dat = mxGetPr(a_mex);
	
	vector<float> a(dim);
	
	for (int i=0; i<dim; i++) {
		a(i) = a_dat[i];
	}
	
	return a;
}

matrix<float> mex2ublas2d(const mxArray *a_mex) {
    const int *mwSize = mxGetDimensions(a_mex);
    int dim0 = dim[0];
    int dim1 = dim[1];
	matrix<float> a(dim0, dim1);
    double *a_dat = mxGetPr(a_mex);
	
	for (int i0=0; i0<dim0; i0++) {
		for (int i1=0; i1<dim1; i1++) {
			a(i0, i1) = a_dat[i0+i1*dim0];
		}
	}
	
	return a;
}

vector<matrix<float> > mex2ublas3d(const mxArray *a_mex) {
    const int *mwSize = mxGetDimensions(a_mex);
	int dim0 = dim[0];
	int dim1 = dim[1];
	int dim2 = dim[2];
	vector<matrix<float> > a(dim2);
	matrix<float> aa(dim0, dim1);
    double *a_dat = mxGetPr(a_mex);
	
	for (int i2=0; i2<dim2; i2++) {
		for (int i1=0; i1<dim1; i1++) {
			for (int i0=0; i0<dim0; i0++) {
				aa(i0, i1) = a_dat[i0+i1*dim0+i2*dim0*dim1];
			}
		}
		a(i2) = aa;
	}
	
	return a;
}

mxArray * ublas2mex1d(vector<float> a) {
	int dim = a.size();
    mxArray *a_mex = mxCreateNumericMatrix(dim, 1, mxDOUBLE_CLASS, mxREAL);
	double *a_dat = mxGetPr(a_mex);
	
	for (int i=0; i<dim; i++) {
		a_dat[i] = a(i);
	}
	
	return a_mex;
}

mxArray * ublas2mex2d(matrix<float> a) {
	int dim1 = a.size1();
	int dim2 = a.size2();
    mxArray *a_mex = mxCreateNumericMatrix(dim1, dim2, mxDOUBLE_CLASS, mxREAL);
    double *a_dat = mxGetPr(a_mex);
	
	for (int i1=0; i1<dim1; i1++) {
		for (int i2=0; i2<dim2; i2++) {
			a_dat[i1+i2*dim1] = a(i1, i2);
		}
	}
	
	return a_mex;
}
