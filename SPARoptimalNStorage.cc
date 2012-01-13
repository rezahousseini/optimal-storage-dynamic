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
	FloatNDArray xc;
	FloatNDArray xd;
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
FloatNDArray C;
FloatNDArray D;
RowVector nul;
RowVector nuc;
RowVector nud;

int32NDArray R;
int32NDArray Rx;
FloatNDArray xc;
FloatNDArray xd;

int32NDArray set_fin;

void init(octave_scalar_map S);
void initOpt(void);
opt_sol solveOpt(float g, float r, float pg, float pr, FloatNDArray pc, FloatNDArray pd, int32NDArray R, FloatNDArray v);
void deleteOpt(void);
int32NDArray randi(int start, int end, int number);
int32NDArray scale(NDArray S, float s);

/* ----------------------------------------------------------------------------*
 * DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T")   *
 * ----------------------------------------------------------------------------*
 * Main function. Call "SPARoptimalNStorage" in Octave.
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

DEFUN_DLD(SPARoptimalNStorage, args, nargout, "rho, g, r, P, S, numI, T")
{
	octave_value_list retval;
	int nargin = args.length();
	if (nargin != 7)
		print_usage();
	else
	{
		rho = args(0).float_value();
		FloatNDArray g = args(1).array_value();
		FloatNDArray r = args(2).array_value();
		octave_scalar_map P = args(3).scalar_map_value();
		octave_scalar_map S = args(4).scalar_map_value();
		int numI = args(5).int_value();
		T = args(6).float_value();
		
		// Prices
		FloatNDArray pg = P.contents("pg").array_value();
		FloatNDArray pr = P.contents("pr").array_value();
		FloatNDArray pc = P.contents("pc").array_value();
		FloatNDArray pd = P.contents("pd").array_value();
		
		numN = g.dims().elem(0);
		numW = g.dims().elem(1);
		
		init(S);
		initOpt();
		
		dim_vector dv_v(numR.sum(0).elem(0), numN, numW);
		
		int32NDArray smpl;
		FloatNDArray v(dv_v, 0); // Value function for the different levels
		int32NDArray NV(dv_v, 0); // Number of visits to the corresponding state
		opt_sol ret, retup, retlo;
		FloatNDArray vhatlo(dim_vector(numSfin, 1));
		FloatNDArray vhatup(dim_vector(numSfin, 1));
		FloatNDArray zlo(dim_vector(numSfin, 1), 0);
		FloatNDArray zup(dim_vector(numSfin, 1), 0);
		float alpha;
		
		FloatNDArray q(dim_vector(numSfin, numN));
		FloatNDArray uc(dim_vector(numS, numN));
		FloatNDArray ud(dim_vector(numS, numN));
		FloatNDArray cost(dim_vector(numN, 1), 0);
		
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
				
				xc.insert(ret.xc, 0, k);
				xd.insert(ret.xd, 0, k);
				
				float Rxerr;
				int count = 0;
				for (int m=0; m<numS; m++)
				{
					if ((int)set_fin(m) == 1)
					{
						// Resource transition function
						Rxerr = nul(m)*(float)R.column(k).elem(count)+T*(nuc(m)*(float)xc.column(k).elem(m)-(1/nud(m))*(float)xd.column(k).elem(m));
						
						if (Rxerr < rho*Qmin(m))
						{
//							printf("Warning - Negative index: Rxerr=%f\n", Rxerr);
							Rx(count, k) = floor(rho*Qmin(m));
						}
						else if (Rxerr > rho*Qmax(m))
						{
//							printf("Warning - Too big index: Rxerr=%f\n", Rxerr);
							Rx(count, k) = floor(rho*Qmax(m));
						}
						else Rx(count, k) = floor(Rxerr);
						
						count = count+1;
					}
				}
				
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
							
							vhatlo(m) = retup.F-ret.F;
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
						
						// Calculate z and insert into v
						v(index+Rx(m, k), k, smpl(k)) = (1-alpha)*v.page(smpl(k)).column(k).elem(index+Rx(m, k))+alpha*vhatlo(m);
						
						if (Rx(m, k)+(octave_int32)1 < numR(m))
						{
							v(index+Rx(m, k)+(octave_int32)1, k, smpl(k)) = (1-alpha)*v.page(smpl(k)).column(k).elem(index+Rx(m, k)+(octave_int32)1)+alpha*vhatup(m);
						}
						
						// Projection operation
						
						// r < Rx
						for (int level=(int)Rx(m, k); level>0; level--)
						{
							if(v((int)index+level-1, k, smpl(k)) >= v((int)index+level, k, smpl(k)))
							{
								break;
							}
							else
							{
//								float vmean = (v((int)index+level-1, k, smpl(k))+((int)Rx(m, k)-level+1)*v((int)index+level, k, smpl(k)))/((int)Rx(m, k)-level+2);
								for (int i=(int)Rx(m, k); i>=level-1; i--)
								{
									v((int)index+i, k, smpl(k)) = vhatlo(m);
								}
							}
						}
						
						// r > Rx
						for (int level=(int)Rx(m, k)+1; level < (int)numR(m); level++)
						{
							if(v((int)index+level+1, k, smpl(k)) <= v((int)index+level, k, smpl(k)))
							{
								break;
							}
							else
							{
//								float vmean = (v((int)index+level+1, k, smpl(k))+(level-(int)Rx(m, k))*v((int)index+level, k, smpl(k)))/(level-(int)Rx(m, k)-1);
								for (int i=(int)Rx(m, k); i<=level+1; i++)
								{
									v((int)index+i, k, smpl(k)) = vhatup(m);
								}
							}
						}
						
						index = index+numR(m);
					} // endfor numSfin
				} // endif
			} // endfor k
		} // endfor iter
		
		// Rescale return value
		for (int k=0; k<numN; k++)
		{
			for (int m=0; m<numS; m++)
			{
				uc(m, k) = xc(m, k)/rho;
				ud(m, k) = xd(m, k)/rho;
				
				cost(k) = cost(k)+uc(m, k)*pc(m, k)+ud(m, k)*pd(m, k);
			}
			cost(k) = cost(k)+g(k)*pg(k)+r(k)*pr(k);
		}
		
		for (int k=0; k<numN; k++)
		{
			for (int m=0; m<numSfin; m++)
			{
				q(m, k) = (float)Rx(m, k)/rho;
			}
		}
		
		retval(0) = octave_value(q);
		retval(1) = octave_value(uc);
		retval(2) = octave_value(ud);
		retval(3) = octave_value(cost);
		
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
	C = S.contents("C").array_value();
	D = S.contents("D").array_value();
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
	xc = FloatNDArray(dim_vector(numS, numN), 0);
	xd = FloatNDArray(dim_vector(numS, numN), 0);
	
	count = 0;
	for (int k=0; k<numS; k++)
	{
		if ((int)set_fin(k) == 1)
		{
			numR(count) = floor(rho*Qmax(k)); // Scale max capacity
			R(count,0) = floor(rho*q0(k)); // Storage level initialization; TODO checking for q0 <= Qmax
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
		glp_set_col_bnds(lp, m, GLP_DB, 0, rho*C(m-1)); // Charge 0 <= uc <= C
		
		sprintf(str, "ud_%i", m);
		glp_set_col_name(lp, numS+m, str);
		glp_set_col_bnds(lp, numS+m, GLP_DB, 0, rho*D(m-1)); // Discharge 0 <= ud <= D
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
		val[m] = nuc(m-1); // uc
		val[numS+m] = -1/nud(m-1); // -ud
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
			
			val[m] = -nuc(m-1); // -uc
			val[numS+m] = 1/nud(m-1); // ud
			
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
			
			val[m] = nuc(m-1); // uc
			val[numS+m] = -1/nud(m-1); // -ud
			
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
 * @param g  Generated energy (node input to satisfie). 1 x 1
 * @param r Requested energy (node output to satisfie). 1 x 1
 * @param pg Price of one amount of generated enery. 1 x 1
 * @param pr Price of one amount of requested enery. 1 x 1
 * @param pc Price vector for charge vaiables of all resources. numS x 1
 * @param pd Price vector for discharge vaiables of all resources. numS x 1
 * @param R Resource level vector for all finite capacities. numSfin x 1
 * @param v Value function approximation for all level steps. sum(numR) x 1
 *
 * @return Optimal solution argument and value.
 *
 */

opt_sol solveOpt(float g, float r, float pg, float pr,
	FloatNDArray pc, FloatNDArray pd, int32NDArray R, FloatNDArray v)
{
	opt_sol retval;
	retval.xc = FloatNDArray(dim_vector(numS, 1));
	retval.xd = FloatNDArray(dim_vector(numS, 1));
	int index = 0;
	int count = 1;
	int ret;
	
	// Objectiv coefficient
	for (int m=1; m<=numS; m++)
	{
		glp_set_obj_coef(lp, m, -rho*pc(m-1)); // -pc*uc
		glp_set_obj_coef(lp, numS+m, -rho*pd(m-1)); // -pd*ud
		
		if ((int)set_fin(m-1) == 1)
		{
			for (int k=1; k<=(int)numR(m-1); k++)
			{
				glp_set_obj_coef(lp, 2*numS+index+k, v(index+k-1));
			}
			index = index+(int)numR(m-1);
			
			// Value function constraint
			// -uc+ud+sum{r = 0..numR-1}ytr = R
			glp_set_row_bnds(lp, 1+count, GLP_FX, (float)R(count-1)*nul(m-1), (float)R(count-1)*nul(m-1));
			
			// Minimum and Maximum capacity constraint
			// Qmin-R <= uc-ud <= Qmax-R
			glp_set_row_bnds(lp, 1+numSfin+count, GLP_DB, (rho*Qmin(m-1)-nul(m-1)*(float)R(count-1))/T,
				(rho*Qmax(m-1)-nul(m-1)*(float)R(count-1))/T);
			
			count = count+1;
		}
	}
	
	// Node balance constraint
	// uc-ud = g-r
	glp_set_row_bnds(lp, 1, GLP_FX, rho*(g-r), rho*(g-r));
	
//	glp_write_lp(lp, NULL, "linearSystem.lp");
//	glp_write_mps(lp, GLP_MPS_FILE, NULL, "linearSystem.mps");
	
	// Solve
	ret = glp_simplex(lp, &parm);
	if (ret != 0)
	{
		printf("No simplex solution. Error %i\n", ret);
	}
	
	retval.F = glp_get_obj_val(lp)+rho*rho*(g*pg+r*pr);
	
	for (int m=1; m<=numS; m++)
	{
		
		retval.xc(m-1) = floor(glp_get_col_prim(lp, m));
		retval.xd(m-1) = floor(glp_get_col_prim(lp, numS+m));
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
