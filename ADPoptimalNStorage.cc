#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <octave/intNDArray.h>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "solver.h"

using namespace boost::numeric::ublas;

vector<float> oct2ublas1d(FloatNDArray a_oct);
matrix<float> oct2ublas2d(FloatNDArray a_oct);
vector<matrix<float> > oct2ublas3d(FloatNDArray a_oct);

/* ----------------------------------------------------------------------------*
 * DEFUN_DLD(ADPoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T,    *
 *  parm")                                                                     *
 * ----------------------------------------------------------------------------*
 * Main function. Call "SPARoptimalNStorage(rho, g, r, P, S, numI, T, parm)" in 
 * Octave.
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
DEFUN_DLD(ADPoptimalNStorage, args, nargout, "rho, g, r, Prices, Storages, numI, T, Parameters") {
	octave_value_list retval;
	
	if (args.length() != 8) print_usage();
	else {
		float rho = 					args(0).float_value();
		FloatNDArray prod = 			args(1).float_array_value();
		FloatNDArray cons = 			args(2).float_array_value();
		octave_scalar_map Prices = 		args(3).scalar_map_value();
		octave_scalar_map Storages = 	args(4).scalar_map_value();
		int numI = 						args(5).int_value();
		float T = 						args(6).float_value();
		octave_scalar_map Parameters = 	args(7).scalar_map_value();
		
		storages S;
		prices P;
		parameters parm;
		
		// Parameters
		parm.gama = Parameters.contents("gamma").float_value();
		parm.alpha0 = Parameters.contents("alpha0").float_value();
		parm.deltaStepMult = Parameters.contents("deltaStepMult").float_value();
		parm.a = Parameters.contents("a").float_value();
		parm.b = Parameters.contents("b").float_value();
		parm.c = Parameters.contents("c").float_value();
		parm.epsilon = Parameters.contents("epsilon").float_value();
		
		matrix<float> g = oct2ublas2d(prod);
		matrix<float> r = oct2ublas2d(cons);
		
		// Prices
		matrix<float> pg = oct2ublas2d(Prices.contents("pg").array_value());
		matrix<float> pr = oct2ublas2d(Prices.contents("pr").array_value());
		vector<matrix<float> > pc = oct2ublas3d(Prices.contents("pc").array_value());
		vector<matrix<float> > pd = oct2ublas3d(Prices.contents("pd").array_value());
		
		S.Qmax = oct2ublas1d(Storages.contents("Qmax").array_value());
		S.Qmin = oct2ublas1d(Storages.contents("Qmin").array_value());
		S.q0 = oct2ublas1d(Storages.contents("q0").array_value());
		S.C = oct2ublas1d(Storages.contents("C").array_value());
		S.D = oct2ublas1d(Storages.contents("D").array_value());
		S.etal = oct2ublas1d(Storages.contents("etal").array_value());
		S.etac = oct2ublas1d(Storages.contents("etac").array_value());
		S.etad = oct2ublas1d(Storages.contents("etad").array_value());
		S.DeltaCmax = oct2ublas1d(Storages.contents("DeltaCmax").array_value());
		S.DeltaDmax = oct2ublas1d(Storages.contents("DeltaDmax").array_value());
		
//		std::cout << S.Qmax << std::endl;
		
//		solution sol = solve(rho, g, r, P, S, numI, T, parm);
		
		// Return values
//		int32NDArray q = FloatNDArray(dim_vector(numSfin, numN));
//		FloatNDArray uc = FloatNDArray(dim_vector(numS, numN));
//		FloatNDArray ud = FloatNDArray(dim_vector(numS, numN));
//		FloatNDArray costIter(dim_vector(numN, 1));
//		FloatNDArray cost(dim_vector(numN, numI));
//		
//		 Rescale return value
//		retval(0) = octave_value(q);
//		retval(1) = octave_value(uc);
//		retval(2) = octave_value(ud);
//		retval(3) = octave_value(cost);
//		retval(4) = octave_value(costIter);
	}
	
	return retval;
}

vector<float> oct2ublas1d(FloatNDArray a_oct) {
	int dim1 = a_oct.dim1();
	int dim2 = a_oct.dim2();
	int dim;
	
	if (dim2 == 1) dim = dim1;
	else dim = dim2;
	
	vector<float> a(dim);
	
	for (int i=0; i<dim; i++) {
			a(i) = a_oct(i);
	}
	
	return a;
}

matrix<float> oct2ublas2d(FloatNDArray a_oct) {
	int dim1 = a_oct.dim1();
	int dim2 = a_oct.dim2();
	matrix<float> a(dim1, dim2);
	
	for (int i1=0; i1<dim1; i1++) {
		for (int i2=0; i2<dim2; i2++) {
			a(i1, i2) = a_oct(i1, i2);
		}
	}
	
	return a;
}

vector<matrix<float> > oct2ublas3d(FloatNDArray a_oct) {
	int dim1 = a_oct.dim1();
	int dim2 = a_oct.dim2();
	int dim3 = a_oct.dim3();
	vector<matrix<float> > a(dim1);
	matrix<float> aa(dim2, dim3);
	
	for (int i1=0; i1<dim1; i1++) {
		for (int i2=0; i2<dim2; i2++) {
			for (int i3=0; i3<dim3; i3++) {
				aa(i2, i3) = a_oct(i1, i2, i3);
			}
		}
		a(i1) = aa;
	}
	
	return a;
}
