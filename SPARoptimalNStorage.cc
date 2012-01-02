#include <octave/oct.h>
#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>
#include <octave/ov-struct.h>
#include <octave/intNDArray.h>
#include <iostream>

using namespace std;

struct opt_sol
{
	float F;
	int32NDArray Rx;
	int32NDArray xc;
	int32NDArray xd;
};

int32NDArray numR;
int numS;
int numSfin;
int numW;
int numN;
glp_prob *lp;
glp_smcp parm;
float T;
float rho;

// Storage parameters
FloatNDArray Qmax;
FloatNDArray Qmin;
FloatNDArray q0;
int32NDArray C;
int32NDArray D;
RowVector nul;
RowVector nuc;
RowVector nud;

int32NDArray R;
int32NDArray Rx;
int32NDArray xc;
int32NDArray xd;

int32NDArray set_fin;

void init(octave_scalar_map S);
void initOpt(void);
opt_sol solveOpt(int g, int r, int pg, int pr, int32NDArray pc, int32NDArray pd, int32NDArray R, FloatNDArray v);
void deleteOpt(void);
int32NDArray randi(int start, int end, int number);
int32NDArray scale(NDArray S, float s);

/* ----------------------------------------------------------------------------*
 * DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho,g,r,P,S,numI,T")         *
 * ----------------------------------------------------------------------------*
 * Main function.
 *
 * @param SPARoptimalNStorage The identifiying name of the program in octave.
 * @param args Input arguments.
 * @param nargout Number of output arguments.
 * @param "rho, g, r, P, S, numI, T" Function call in octave.
 *
 * @return q, uc, ud
 *
 */ 

DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T")
{
	octave_value_list retval;
	int nargin = args.length();
	if (nargin != 7)
		print_usage();
	else
	{
		rho = args(0).float_value();
		int32NDArray g = scale(args(1).array_value(), rho);
		int32NDArray r = scale(args(2).array_value(), rho);
		octave_scalar_map P = args(3).scalar_map_value();
		octave_scalar_map S = args(4).scalar_map_value();
		int numI = args(5).int_value();
		T = args(6).float_value();
		
		// Prices
		int32NDArray pg = scale(P.contents("pg").array_value(), rho);
		int32NDArray pr = scale(P.contents("pr").array_value(), rho);
		int32NDArray pc = scale(P.contents("pc").array_value(), rho);
		int32NDArray pd = scale(P.contents("pd").array_value(), rho);
		
		numN = g.dims().elem(0);
		numW = g.dims().elem(1);
		
		init(S);
		initOpt();
		
		dim_vector dv_v(numR.sum(0).elem(0), numN, numW);
		
		int32NDArray smpl;
		FloatNDArray v(dv_v, 0); // Value function for the different levels
		int32NDArray NV(dv_v, 1); // Number of visits to the corresponding state
		opt_sol ret, retlo, retup;
		FloatNDArray vhatlo(dim_vector(numSfin, 1));
		FloatNDArray vhatup(dim_vector(numSfin, 1));
		float alpha;
		FloatNDArray zlo(dim_vector(numSfin, 1), 0);
		FloatNDArray zup(dim_vector(numSfin, 1), 0);
		
		FloatNDArray q(dim_vector(numSfin, numN));
		FloatNDArray uc(dim_vector(numS, numN));
		FloatNDArray ud(dim_vector(numS, numN));
		
		for (int i=0; i<numI; i++)
		{
			smpl = randi(0,numW,numN); // Generate sample
			
			for (int k=0; k<numN; k++)
			{
				// Compute pre-decision asset level
				if (k != 0) R.insert(Rx.column(k-1), 0, k);
				
				// Find optimal value function and 
				// compute post-decision asset level
				ret = solveOpt(
					g.column(smpl(k)).elem(k), r.column(smpl(k)).elem(k),
					pg.column(smpl(k)).elem(k), pr.column(smpl(k)).elem(k),
					pc.page(smpl(k)).column(k), pd.page(smpl(k)).column(k),
					R.column(k), v.page(smpl(k)).column(k)
				);
				
				Rx.insert(ret.Rx, 0, k);
				xc.insert(ret.xc, 0, k);
				xd.insert(ret.xd, 0, k);
				
				// Count number of visits
				octave_int32 index = 0;
				for (int m=0; m<numSfin; m++)
				{
					NV(index+Rx(m, k), k, smpl(k)) = NV.page(smpl(k)).column(k).elem(index+Rx(m, k))+(octave_int32)1;
					index = index+numR(m);
				}
				
				if (k < numN-1)
				{
					// Observe sample slopes
					ret = solveOpt(
						g.column(smpl(k+1)).elem(k+1),r.column(smpl(k+1)).elem(k+1),
						pg.column(smpl(k+1)).elem(k+1),pr.column(smpl(k+1)).elem(k+1),
						pc.page(smpl(k+1)).column(k+1),pd.page(smpl(k+1)).column(k+1),
						Rx.column(k),v.page(smpl(k+1)).column(k+1)
					);
					
					int32NDArray Rxlo(dim_vector(numSfin,1));
					int32NDArray Rxup(dim_vector(numSfin,1));
					
					index = 0;
					for (int m=0; m<numSfin; m++)
					{
						Rxlo.insert(Rx.column(k), 0, 0);
						Rxup.insert(Rx.column(k), 0, 0);
						
						if (Rx(m,k) == (octave_int32)0)
						{
							Rxup(m) = Rx(m,k)+(octave_int32)1;
							
							retup = solveOpt(
								g.column(smpl(k+1)).elem(k+1), r.column(smpl(k+1)).elem(k+1),
								pg.column(smpl(k+1)).elem(k+1), pr.column(smpl(k+1)).elem(k+1),
								pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
								Rxup, v.page(smpl(k+1)).column(k+1)
							);
							
							vhatlo(m) = (octave_int32)0;
//							vhatlo(m) = ret.F;
							vhatup(m) = retup.F-ret.F;
						}
						else if (Rx(m,k) == numR(m)-(octave_int32)1)
						{
							Rxlo(m) = Rx(m, k)-(octave_int32)1;
							
							retlo = solveOpt(
								g.column(smpl(k+1)).elem(k+1), r.column(smpl(k+1)).elem(k+1),
								pg.column(smpl(k+1)).elem(k+1), pr.column(smpl(k+1)).elem(k+1),
								pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
								Rxlo, v.page(smpl(k+1)).column(k+1)
							);
							
							vhatlo(m) = ret.F-retlo.F;
							vhatup(m) = (octave_int32)0;
						}
						else
						{
							Rxlo(m) = Rx(m,k)-(octave_int32)1;
							Rxup(m) = Rx(m,k)+(octave_int32)1;
							
							retlo = solveOpt(
								g.column(smpl(k+1)).elem(k+1), r.column(smpl(k+1)).elem(k+1),
								pg.column(smpl(k+1)).elem(k+1), pr.column(smpl(k+1)).elem(k+1),
								pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
								Rxlo, v.page(smpl(k+1)).column(k+1)
							);
							retup = solveOpt(
								g.column(smpl(k+1)).elem(k+1), r.column(smpl(k+1)).elem(k+1),
								pg.column(smpl(k+1)).elem(k+1), pr.column(smpl(k+1)).elem(k+1),
								pc.page(smpl(k+1)).column(k+1), pd.page(smpl(k+1)).column(k+1),
								Rxup, v.page(smpl(k+1)).column(k+1)
							);
							
							vhatlo(m) = ret.F-retlo.F;
							vhatup(m) = retup.F-ret.F;
						}
						
						// Update slopes
						// Calculate alpha
						alpha = 1/(float)NV.page(smpl(k)).column(k).elem(index+Rx(m, k));
						
						// Calculate z
						zlo(m) = (1-alpha)*v.page(smpl(k)).column(k).elem(index+Rx(m, k))+alpha*vhatlo(m);
						
						if (Rx(m, k)+(octave_int32)1 < numR(m)-(octave_int32)1)
						{
							zup(m) = (1-alpha)*v.page(smpl(k)).column(k).elem(index+Rx(m, k)+(octave_int32)1)+alpha*vhatup(m);
						}
						else
						{
							zup(m) = vhatup(m);
						}
						
						// Projection operation
						v(index+Rx(m, k),k,smpl(k)) = zlo(m);
						
						if (Rx(m, k)+(octave_int32)1 < numR(m)-(octave_int32)1)
						{
							v(index+Rx(m, k)+(octave_int32)1,k,smpl(k)) = zup(m);
						}
						
						for (octave_int32 level=0; level<numR(m); level=level+(octave_int32)1)
						{
							if (level < Rx(m, k) and 
								v.page(smpl(k)).column(k).elem(index+level) <= zlo(m))
							{
								v(index+level, k, smpl(k)) = zlo(m);
							}
							else if (level > (Rx(m, k)+(octave_int32)1) and 
								v.page(smpl(k)).column(k).elem(index+level) >= zup(m))
							{
								v(index+level, k, smpl(k)) = zup(m);
							}
						} // endfor level
						
						index = index+numR(m);
					} // endfor numSfin
				} // endif
			} // endfor k
		} // endfor iter
		
		// Rescale return value
		for (int m=0; m<numS; m++)
		{
			for (int k=0; k<numN; k++)
			{
				uc(m, k) = ((float)xc(m, k)+1)/rho;
				ud(m, k) = ((float)xd(m, k)+1)/rho;
			}
		}
		
		for (int m=0; m<numSfin; m++)
		{
			for (int k=0; k<numN; k++)
			{
				q(m, k) = ((float)R(m, k)+1)/rho;
			}
		}
		
		retval(0) = octave_value(q);
		retval(1) = octave_value(uc);
		retval(2) = octave_value(ud);
		
		deleteOpt();
	}
return retval;
}

/* ----------------------------------------------------------------------------*
 * void init(octave_scalar_map S)                                              *
 * ----------------------------------------------------------------------------*
 * Initiation of the algorithm.
 *
 * @param S Structure with storage parameters.
 *
 * @return void
 *
 */ 

void init(octave_scalar_map S)
{
	// Storage parameters
	Qmax = S.contents("Qmax").array_value();
	Qmin = S.contents("Qmin").array_value();
	q0 = S.contents("q0").array_value();
	C = scale(S.contents("C").array_value(), rho);
	D = scale(S.contents("D").array_value(), rho);
	nul = S.contents("nul").row_vector_value();
	nuc = S.contents("nuc").row_vector_value();
	nud = S.contents("nud").row_vector_value();
	
	// Number of ressources
	numS = Qmax.nelem();
	set_fin = int32NDArray(dim_vector(numS, 1), 0);
	
	int count = 0;
	for (int k=0; k<numS; k++)
	{
		if (Qmax(k) != 1.0/0.0) // Qmax < inf
		{
			set_fin(k) = 1;
		}
	}
	
	numSfin = set_fin.sum(0).elem(0);
	numR = int32NDArray(dim_vector(numSfin,1));
	
	R = int32NDArray(dim_vector(numSfin, numN), 0); // Pre-decision asset level
	Rx = int32NDArray(dim_vector(numSfin, numN), 0); // Post-decision asset level
	xc = int32NDArray(dim_vector(numS, numN), 0);
	xd = int32NDArray(dim_vector(numS, numN), 0);
	
	count = 0;
	for (int k=0; k<numS; k++)
	{
		if ((int)set_fin(k) == 1)
		{
			numR(count) = floor(rho*Qmax(k)); // Scale max capacity
			R(count,0) = floor(rho*q0(k))-1; // Storage level initialization; TODO checking for q0 <= Qmax
			count = count+1;
		}
	}
	
	srand(time(NULL));
}

/* ----------------------------------------------------------------------------*
 * void initOpt(void)                                                          *
 * ----------------------------------------------------------------------------*
 * Initiat the linear programming problem.
 *
 * @param void
 *
 * @return void
 *
 */

void initOpt(void)
{
	lp = glp_create_prob();
	int numV = (int)numR.sum(0).elem(0);
	glp_add_rows(lp, 1+2*numSfin);
	glp_add_cols(lp, 2*numS+numV);
	glp_set_obj_name(lp, "profit");
	glp_set_obj_dir(lp, GLP_MAX);
	int ind[2*numS+numV+1];
	double val[2*numS+numV+1];
	int count;
	char str[80];
	int index;
	
	for (int k=1; k<=2*numS+numV; k++)
	{
		ind[k] = k; // ind and val start at 1 not at 0!!!!
	}
	
	// Structural variable bounds
	for (int m=1; m<=numS; m++)
	{
		sprintf(str, "uc_%i", m);
		glp_set_col_name(lp, m, str);
		glp_set_col_bnds(lp, m, GLP_DB, 0, C(m-1)); // Charge 0 <= uc <= C
		
		sprintf(str, "ud_%i", m);
		glp_set_col_name(lp, numS+m, str);
		glp_set_col_bnds(lp, numS+m, GLP_DB, 0, D(m-1)); // Discharge 0 <= ud <= D
	}
	index = 0;
	for (int m=0; m<numSfin; m++)
	{
		for (int k=1; k<=(int)numR(m); k++)
		{
			sprintf(str, "y_%i_%i", m+1, k);
			glp_set_col_name(lp, 2*numS+index+k, str);
			glp_set_col_bnds(lp, 2*numS+index+k, GLP_DB, 0, 1); // 0 <= ytr <= 1
		}
		index = index+(int)numR(m);
	}
	
	// Node balance constraint
	// uc-ud = g-r
	glp_set_row_name(lp, 1, "balance");
	for (int m=1; m<=numS; m++)
	{
		val[m] = 1; // uc
		val[numS+m] = -1; // -ud
	}
	for (int m=1; m<=numV; m++)
	{
		val[2*numS+m] = 0;
	}
	glp_set_mat_row(lp, 1, 2*numS+numV, ind, val);
	
	// Value function constraint
	// -uc+ud+sum{i in numR}ytr = R
	count = 0;
	index = 0;
	for (int m=1; m<=numS; m++)
	{
		if ((int)set_fin(m-1) == 1)
		{
			// Reset the val vector to 0
			for (int n=1; n<=2*numS+numV; n++)
			{
				val[n] = 0;
			}
			
			val[m] = -1; // -uc
			val[numS+m] = 1; // ud
			
			for (int k=1; k<=(int)numR(count); k++)
			{
				val[2*numS+index+k] = 1;
			}
			index = index+(int)numR(count);
			
			sprintf(str, "value_function_%i", count+1);
			glp_set_row_name(lp, 1+count+1, str);
			glp_set_mat_row(lp, 1+count+1, 2*numS+numV, ind, val);
			count = count+1;
		}
	}
	
	// Minimum and Maximum capacity constraint
	// Qmin-R <= uc-ud <= Qmax-R
	count = 1;
	for (int m=1; m<=numS; m++)
	{
		if ((int)set_fin(m-1) == 1)
		{
			// Reset the val vector to 0
			for (int n=1; n<=2*numS+numV; n++)
			{
				val[n] = 0;
			}
			
			val[m] = 1; // uc
			val[numS+m] = -1; // -ud
			
			sprintf(str, "capacity_bound_%i", count);
			glp_set_row_name(lp, 1+numSfin+count, str);
			glp_set_mat_row(lp, 1+numSfin+count, 2*numS+numV, ind, val);
			count = count+1;
		}
	}
	
	// Display errors and warnings
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_ERR;
}

/* --------------------------------------------------------------------------- *
 * opt_sol solveOpt(int g, int r, int pg, int pr,                              *
 *  int32NDArray pc, int32NDArray pd, int32NDArray R, FloatNDArray v)          *
 * --------------------------------------------------------------------------- *
 * Solving the linear programming problem.
 *
 *    | Description                                             | Size
 * ----------------------------------------------------------------------------
 * g  | Generated energy (node input to satisfie).              | 1 x 1
 * r  | Requested energy (node output to satisfie).             | 1 x 1
 * pg | Price of one amount of generated enery.                 | 1 x 1
 * pr | Price of one amount of requested enery.                 | 1 x 1
 * pc | Price vector for charge vaiables of all resources.      | numS x 1
 * pd | Price vector for discharge vaiables of all resources.   | numS x 1
 * R  | Resource level vector for all the finite capacities.    | numSfin x 1
 * v  | Piecewise value function approximation for every level  | sum(numR) x 1
 *    | step.                                                   |
 *
 */

opt_sol solveOpt(int g, int r, int pg, int pr,
	int32NDArray pc, int32NDArray pd, int32NDArray R, FloatNDArray v)
{
	opt_sol retval;
	retval.Rx = int32NDArray(dim_vector(numSfin, 1));
	retval.xc = int32NDArray(dim_vector(numS, 1));
	retval.xd = int32NDArray(dim_vector(numS, 1));
	int index = 0;
	int count = 1;
	int ret;
	int32NDArray Rxerr(dim_vector(numSfin, 1));
	
	// Objectiv coefficient
	for (int m=1; m<=numS; m++)
	{
		glp_set_obj_coef(lp, m, -pc(m-1)); // -pc*uc
		glp_set_obj_coef(lp, numS+m, -pd(m-1)); // -pd*ud
		
		if ((int)set_fin(m-1) == 1)
		{
			for (int k=1; k<=(int)numR(m-1); k++)
			{
				glp_set_obj_coef(lp, 2*numS+index+k, v(index+k-1));
			}
			index = index+(int)numR(m-1);
			
			// Value function constraint
			// -uc+ud+sum{r = 0..numR-1}ytr = R
			glp_set_row_bnds(lp, 1+count, GLP_FX, (int)R(count-1)+1, (int)R(count-1)+1);
			
			// Minimum and Maximum capacity constraint
			// Qmin-R <= uc-ud <= Qmax-R
			glp_set_row_bnds(lp, 1+numSfin+count, GLP_DB, floor(rho*(float)Qmin(m-1))-(int)R(count-1)+1, (int)numR(count-1)-(int)R(count-1)+1);
			
			count = count+1;
		}
	}
	
	// Node balance constraint
	// uc-ud = g-r
	glp_set_row_bnds(lp, 1, GLP_FX, g-r, g-r);
	
//	glp_write_lp(lp, NULL, "linearSystem.lp");
//	glp_write_mps(lp, GLP_MPS_FILE, NULL, "linearSystem.mps");
	
	// Solve
	ret = glp_simplex(lp, &parm);
	if (ret != 0)
	{
		printf("No simplex solution. Error %i\n", ret);
	}
	
	retval.F = glp_get_obj_val(lp)+g*pg+r*pr;
	
	count = 0;
	for (int m=1; m<=numS; m++)
	{
		
		retval.xc(m-1) = floor(glp_get_col_prim(lp, m));
		retval.xd(m-1) = floor(glp_get_col_prim(lp, numS+m));
		
		if ((int)set_fin(m-1) == 1)
		{
			Rxerr(count) = floor(nul(m-1)*(float)R(count)+T*(nuc(m-1)*glp_get_col_prim(lp, m)-(1/nud(m-1))*glp_get_col_prim(lp, numS+m)));
			
			if ((int)Rxerr(count) < 0)
			{
				retval.Rx(count) = 0;
				printf("solveOpt Warning: Negative index: Rxerr=%i\n",(int)Rxerr(count));
			}
			else if ((int)Rxerr(count) > (int)numR(count)-1)
			{
				retval.Rx(count) = (int)numR(count)-1;
				printf("solveOpt Warning: Too big index: Rxerr=%i\n",(int)Rxerr(count));
			}
			else retval.Rx(count) = Rxerr(count);
			
			count = count+1;
		}
	}
	
	return retval;
}

/* ----------------------------------------------------------------------------*
 * void deleteOpt(void)                                                        *
 * ----------------------------------------------------------------------------*
 * Deleting the linear programming problem.
 *
 * @param void
 *
 * @return void
 *
 */

void deleteOpt(void)
{
	glp_delete_prob(lp);
}

/* ----------------------------------------------------------------------------*
 * int32NDArray randi(int min, int max, int length)                            *
 * ----------------------------------------------------------------------------*
 * Generating random integer vector.
 *
 * @param min Minimal number of random vector.
 * @param max Maximal number of random vector.
 * @param length Length of the random integer vector.
 *
 * @return Vector with length <length> random integers between <min> and <max>.
 *
 */

int32NDArray randi(int min, int max, int length)
{
	dim_vector dv;
	dv(0) = length;
	dv(1) = 1;
	int32NDArray sample(dv);
	
	for (int k=0; k<length; k++)
	{
		sample(k) = floor((min+(max+1-min)*rand()/RAND_MAX));
	}
	
	return sample;
}

/* ----------------------------------------------------------------------------*
 * int32NDArray scale(NDArray S, float s)                                      *
 * ----------------------------------------------------------------------------*
 * Scaling an array by a float number.
 *
 * @param S Array to scale.
 * @param s Scale factor.
 *
 * @return Array of integers scaled with factor s.
 *
 */

int32NDArray scale(NDArray S, float s)
{
	dim_vector dv = S.dims();
	int32NDArray S_int(dv);
	
	for (int k=0; k<dv(0); k++)
	{
		for (int m=0; m<dv(1); m++)
		{
			S_int(k,m) = floor(s*S(k,m));
		}
	}
	
	return S_int;
}
