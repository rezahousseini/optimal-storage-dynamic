#include <octave/oct.h>
#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>
#include <octave/ov-struct.h>
#include <octave/int8NDArray.h>
#include <iostream>

using namespace std;

struct optimal_value
{
	float F;
	int Rx;
};

glp_prob *lp;
glp_smcp parm;

void initOpt(octave_scalar_map S, float T, int BR);
optimal_value solveOpt(float g, float p, float r, int R, RowVector v, float T, octave_scalar_map S);
void deleteOpt(void);
intNDArray<int> randi(int start, int end, int number);

DEFUN_DLD(SPARoptimal1StorageV1_0, args, nargout, "W,S,Iter,BR,T")
{
	octave_value_list retval;
	int nargin = args.length();
	if (nargin != 5)
		print_usage();
	else
	{
		srand(time(NULL));
		octave_scalar_map W = args(0).scalar_map_value();
		octave_scalar_map S = args(1).scalar_map_value();
		int Iter = args(2).int_value();
		int BR = args(3).int_value();
		float T = args(3).float_value();

		Matrix g = W.contents(0).matrix_value();
		Matrix p = W.contents(1).matrix_value();
		Matrix r = W.contents(2).matrix_value();

		dim_vector dv_g = g.dims();
		int BW = dv_g(0);
		int N = dv_g(1);

		dim_vector dv_R;
		dv_R(0) = N;
		dv_R(1) = 1;

		Matrix v(BW*N,BR);
		intNDArray<int> R(dv_R,0); // Pre-decision asset level
		intNDArray<int> Rx(dv_R,0); // Post-decision asset level
		Matrix NV(BW*N,BR); // Number of visits to the corresponding state
		float rho = S.contents(0).float_value()*(BR-1); // 1/nu
		int R0 = floor((S.contents(1).float_value()*rho)); // Storage level initialization
		optimal_value ret, ret1, ret2;
		float vhatlo, vhatup;
		Matrix z(BW,BR);
		Matrix alpha(BW,BR);
		intNDArray<int> sample;
		RowVector RxRet(N);

		v.fill(0);
		NV.fill(0);

		initOpt(S, T, BR);

		for(int iter=0; iter<Iter; iter++)
		{
			// Generate sample
			sample = randi(0,BW,N);

			for(int k=0; k<N; k++)
			{

				// Compute pre-decision asset level
				if(k == 0) R(k) = R0;
				else R(k) = Rx(k-1);

				// Find optimal value function and compute post-decision asset level
				ret = solveOpt(g(sample(k),k),p(sample(k),k),r(sample(k),k),R(k),v.row(sample(k)*N+k),T,S);
				Rx(k) = ret.Rx;

				// Count number of visits
				NV(sample(k)*N+k,Rx(k)) = NV(sample(k)*N+k,Rx(k))+1;

				if(k < N-1)
				{
					// Observe sample slopes
					ret1 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k),v.row(sample(k+1)*N+k+1),T,S);
					if(Rx(k) == 0) ret2.F = 0;
					else
					{
						ret2 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k)-1,v.row(sample(k+1)*N+k+1),T,S);
					}
					vhatlo = ret1.F-ret2.F;

					ret2 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k),v.row(sample(k+1)*N+k+1),T,S);
					if(Rx(k) == BR-1) ret1.F = ret2.F;
					else
					{
						ret1 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k)+1,v.row(sample(k+1)*N+k+1),T,S);
					}
					vhatup = ret1.F-ret2.F;

					// Update slopes
					z.fill(0);
					alpha.fill(0);
					for(int state=0; state<BW; state++)
					{
						for(int level=0; level<BR; level++)
						{
							if(state == sample(k))
							{
								// Calculate alpha and z
								if(level == Rx(k))
								{
									RowVector NVmax(BW);
									for(int l=0; l<BW; l++)
									{
										NVmax(l) = NV.row(l*N+k).max();
									}
									alpha(state,level) = 1/NVmax.max();
									z(state,level) = (1-alpha(state,level))*v(state*N+k,level)+alpha(state,level)*vhatlo;
								}
								else if(level == Rx(k)+1)
								{
									RowVector NVmax(BW);
									for(int l=0; l<BW; l++)
									{
										NVmax(l) = NV.row(l*N+k).max();
									}
									alpha(state,level) = 1/NVmax.max();
									z(state,level) = (1-alpha(state,level))*v(state*N+k,level)+alpha(state,level)*vhatup;
								}
								else
								{
									alpha(state,level) = 0;
									z(state,level) = v(state*N+k,level);
								}

								// Projection operation
								if(level < Rx(k) and z(state,level) <= z(sample(k),Rx(k)))
								{
									v(sample(k)*N+k,level)=z(sample(k),Rx(k));
								}
								else if(level > (Rx(k)+1) and z(state,level) >= z(sample(k),(Rx(k)+1)))
								{
									v(sample(k)*N+k,level)=z(sample(k),Rx(k)+1);
								}
								else
								{
									v(sample(k)*N+k,level)=z(state,level);
								}
							}
						} // endfor level
					} // endfor state
				} // endif
			} // endfor k
		} // endfor iter
		for(int k=0; k<N; k++)
		{
			RxRet(k) = Rx(k)/rho;
		}

		retval(0) = octave_value(RxRet);
		deleteOpt();
	}
return retval;
}

void deleteOpt(void)
{
	glp_delete_prob(lp);
}

void initOpt(octave_scalar_map S, float T, int BR)
{
	float q0 = S.contents(1).float_value();
	float C = S.contents(2).float_value();
	float D = S.contents(3).float_value();
	float nul = S.contents(4).float_value();
	float nuc = S.contents(5).float_value();
	float nud = S.contents(6).float_value();

	int ind[4+BR];
	double val[4+BR];

	lp = glp_create_prob();
	glp_add_rows(lp, 3);
	glp_add_cols(lp, 4+BR);
	glp_set_obj_dir(lp, GLP_MAX);

	glp_set_col_bnds(lp, 1, GLP_DB, 0, C); // Storage charge uc
	glp_set_col_bnds(lp, 2, GLP_DB, 0, D); // Storage discharge ud
	glp_set_col_bnds(lp, 3, GLP_LO, 0, 0); // Grid import energy ggr
	glp_set_col_bnds(lp, 4, GLP_LO, 0, 0); // Grid export energy rgr
	for(int m=1; m<=BR; m++)
	{
		glp_set_col_bnds(lp, m+4, GLP_DB, 0, 1);
	}

	// Objectiv coefficient
	glp_set_obj_coef(lp, 1, 0);
	glp_set_obj_coef(lp, 2, 0);
	glp_set_obj_coef(lp, 4, 0);

	ind[1] = 1; val[1] = 1;
	ind[2] = 2; val[2] = -1;
	ind[3] = 3; val[3] = -1;
	ind[4] = 4; val[4] = 1;
	for(int m=1; m<=BR; m++)
	{
		ind[m+4] = m+4; val[m+4] = 0;
	}
	glp_set_mat_row(lp, 1, 4+BR, ind, val);

	ind[1] = 1; val[1] = -1;
	ind[2] = 2; val[2] = 1;
	ind[3] = 3; val[3] = 0;
	ind[4] = 4; val[4] = 0;
	for(int m=1; m<=BR; m++)
	{
		ind[m+4] = m+4; val[m+4] = 1;
	}
	glp_set_mat_row(lp, 2, 4+BR, ind, val);

	ind[1] = 1; val[1] = 1;
	ind[2] = 2; val[2] = -1;
	ind[3] = 3; val[3] = 0;
	ind[4] = 4; val[4] = 0;
	for(int m=1; m<=BR; m++)
	{
		ind[m+4] = m+4; val[m+4] = 0;
	}
	glp_set_mat_row(lp, 3, 4+BR, ind, val);

	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_ERR;
}

optimal_value solveOpt(float g, float p, float r, int R, RowVector v, float T, octave_scalar_map S)
{
	float Qmax = S.contents(0).float_value();

	int BR = (int)v.nelem();
	float rho = Qmax*(BR-1); // 1/nu
	float q = (float)R/rho;
	optimal_value retval;

	glp_set_obj_coef(lp, 3, -p);
	for(int m=1; m<=BR; m++)
	{
		glp_set_obj_coef(lp, m+4, v(m-1));
	}

	glp_set_row_bnds(lp, 1, GLP_FX, g-r, g-r);

	glp_set_row_bnds(lp, 2, GLP_UP, 0, q);

	glp_set_row_bnds(lp, 3, GLP_UP, 0, Qmax-q);

	glp_simplex(lp, &parm);

	float uc = glp_get_col_prim(lp, 1);
	float ud = glp_get_col_prim(lp, 2);
	float ggr = glp_get_col_prim(lp, 3);
	float rgr = glp_get_col_prim(lp, 4);

	retval.F = glp_get_obj_val(lp);
	float err = (q+uc-ud)*rho;
	if (err < 0)
	{
		retval.Rx = 0;
		printf("solveOpt Warning: Negative index: err=%g\n",err);
	}
	else if(err > BR-1)
	{
		retval.Rx = BR-1;
		printf("solveOpt Warning: Too big index: err=%g\n",err);
	}
	else retval.Rx = floor(err);

	return retval;
}

intNDArray<int> randi(int start, int end, int number)
{
	dim_vector dv;
	dv(0) = number;
	dv(1) = 1;
	intNDArray<int> sample(dv);

	for(int k=0; k<number; k++)
	{
		sample(k) = floor(rand()%end+start);
	}

	return sample;
}
