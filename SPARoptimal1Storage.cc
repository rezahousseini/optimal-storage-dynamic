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

int BW;
int BR;
int N;
glp_prob *lp;
glp_smcp parm;
float T;
float rho;

// Storage parameters
float Qmax;
float Qmin;
float q0;
float C;
float D;
float nul;
float nuc;
float nud;

void initOpt(void);
optimal_value solveOpt(float g, float p, float r, int R, RowVector v);
void deleteOpt(void);
intNDArray<int> randi(int start, int end, int number);

DEFUN_DLD(SPARoptimal1StorageV1_1, args, nargout, "W,S,Iter,BR,T")
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
		BR = args(3).int_value();
		T = args(4).float_value();

		Matrix g = W.contents(0).matrix_value();
		Matrix p = W.contents(1).matrix_value();
		Matrix r = W.contents(2).matrix_value();

		Qmax = S.contents("Qmax").float_value();
		Qmin = S.contents("Qmin").float_value();
		q0 = S.contents("q0").float_value();
		C = S.contents("C").float_value();
		D = S.contents("D").float_value();
		nul = S.contents("nul").float_value();
		nuc = S.contents("nuc").float_value();
		nud = S.contents("nud").float_value();

		dim_vector dv_g = g.dims();
		BW = dv_g(0);
		N = dv_g(1);

		dim_vector dv_R;
		dv_R(0) = N;
		dv_R(1) = 1;

		Matrix v(BW*N,BR-1);
		intNDArray<int> R(dv_R,0); // Pre-decision asset level
		intNDArray<int> Rx(dv_R,0); // Post-decision asset level
		Matrix NV(BW*N,BR-1); // Number of visits to the corresponding state
		rho = (BR-2)/Qmax; // 1/nu
		int R0 = floor(q0*rho); // Storage level initialization
		optimal_value ret, ret1, ret2;
		float vhatlo, vhatup;
		Matrix z(1,2);
		float alpha;
		intNDArray<int> sample;
		RowVector q(N);

		v.fill(0);
		NV.fill(0);
		z.fill(0);

		initOpt();

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
				ret = solveOpt(g(sample(k),k),p(sample(k),k),r(sample(k),k),R(k),v.row(sample(k)*N+k));
				Rx(k) = ret.Rx;

				// Count number of visits
				NV(sample(k)*N+k,Rx(k)) = NV(sample(k)*N+k,Rx(k))+1;

				if(k < N-1)
				{
					// Observe sample slopes
					ret1 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k),v.row(sample(k+1)*N+k+1));
					if(Rx(k) == 0) vhatlo = ret1.F;
					else
					{
						ret2 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k)-1,v.row(sample(k+1)*N+k+1));
						vhatlo = ret1.F-ret2.F;
					}

					ret2 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k),v.row(sample(k+1)*N+k+1));
					if(Rx(k) == BR-1) vhatup = ret2.F;
					else
					{
						ret1 = solveOpt(g(sample(k+1),k+1),p(sample(k+1),k+1),r(sample(k+1),k+1),Rx(k)+1,v.row(sample(k+1)*N+k+1));
						vhatup = ret1.F-ret2.F;
					}

					// Update slopes
					// Calculate alpha
					RowVector NVmax(BW);
					for(int l=0; l<BW; l++)
					{
						NVmax(l) = NV.row(l*N+k).max();
					}
					alpha = 1/NVmax.max();

					// Calculate z
					z(0) = (1-alpha)*v(sample(k)*N+k,Rx(k))+alpha*vhatlo;
					if(Rx(k)+1 < BR-1)
					{
						z(1) = (1-alpha)*v(sample(k)*N+k,Rx(k)+1)+alpha*vhatup;
					}
					else
					{
						z(1) = vhatup;
					}

					// Projection operation
					for(int level=0; level<BR-1; level++)
					{
						if(level < Rx(k) and v(sample(k)*N+k,level) <= z(0))
						{
							v(sample(k)*N+k,level)=z(0);
						}
						else if(level > (Rx(k)+1) and v(sample(k)*N+k,level) >= z(1))
						{
							v(sample(k)*N+k,level)=z(1);
						}
						else if(level == Rx(k))
						{
							v(sample(k)*N+k,level)=z(0);
						}
						else if(level == Rx(k)+1)
						{
							v(sample(k)*N+k,level)=z(1);
						}
					} // endfor level
				} // endif
			} // endfor k
		} // endfor iter
		for(int k=0; k<N; k++)
		{
			q(k) = R(k)/rho;
		}

		retval(0) = octave_value(q);
retval(1) = octave_value(R);

		deleteOpt();
	}
return retval;
}

void deleteOpt(void)
{
	glp_delete_prob(lp);
}

void initOpt(void)
{
	int ind[4+BR-1];
	double val[4+BR-1];

	lp = glp_create_prob();
	glp_add_rows(lp, 4);
	glp_add_cols(lp, 4+BR-1);
	glp_set_obj_dir(lp, GLP_MAX);

	// Structural variable bounds
	glp_set_col_bnds(lp, 1, GLP_LO, 0, 0); // Storage charge uc >= 0
	glp_set_col_bnds(lp, 2, GLP_LO, 0, 0); // Storage discharge ud >= 0
	glp_set_col_bnds(lp, 3, GLP_LO, 0, 0); // Grid import energy ggr >= 0
	glp_set_col_bnds(lp, 4, GLP_LO, 0, 0); // Grid export energy rgr >= 0
	for(int m=1; m<=BR-1; m++)
	{
		glp_set_col_bnds(lp, m+4, GLP_DB, 0, 1); // 0 <= ytr <= 1
	}

	// Objectiv coefficient
	glp_set_obj_coef(lp, 1, 0); // 0*uc
	glp_set_obj_coef(lp, 2, 0); // 0*ud
	glp_set_obj_coef(lp, 4, 0); // 0*rgr

	// Node balance constraint
	// uc-ud-ggr+rgr = g-r
	ind[1] = 1; val[1] = 1;
	ind[2] = 2; val[2] = -1;
	ind[3] = 3; val[3] = -1;
	ind[4] = 4; val[4] = 1;
	for(int m=1; m<=BR-1; m++)
	{
		ind[m+4] = m+4; val[m+4] = 0;
	}
	glp_set_mat_row(lp, 1, 4+BR-1, ind, val);

	// Value function constraint
	// -T*nuc*uc*rho+T*(1/nud)*ud*rho+sum{i in BR}ytr = nul*R
	ind[1] = 1; val[1] = -T*nuc*rho;
	ind[2] = 2; val[2] = T*(1/nud)*rho;
	ind[3] = 3; val[3] = 0;
	ind[4] = 4; val[4] = 0;
	for(int m=1; m<=BR-1; m++)
	{
		ind[m+4] = m+4; val[m+4] = 1;
	}
	glp_set_mat_row(lp, 2, 4+BR-1, ind, val);

	// Maximum capacity constraint
	// T*nuc*uc-T*(1/nud)*ud <= Qmax-nul*q
	ind[1] = 1; val[1] = T*nuc;
	ind[2] = 2; val[2] = -T*(1/nud);
	ind[3] = 3; val[3] = 0;
	ind[4] = 4; val[4] = 0;
	for(int m=1; m<=BR-1; m++)
	{
		ind[m+4] = m+4; val[m+4] = 0;
	}
	glp_set_mat_row(lp, 3, 4+BR-1, ind, val);

	// Minimum capacity constraint
	// T*nuc*uc-T*(1/nud)*ud >= Qmin-nul*q
	ind[1] = 1; val[1] = T*nuc;
	ind[2] = 2; val[2] = -T*(1/nud);
	ind[3] = 3; val[3] = 0;
	ind[4] = 4; val[4] = 0;
	for(int m=1; m<=BR-1; m++)
	{
		ind[m+4] = m+4; val[m+4] = 0;
	}
	glp_set_mat_row(lp, 3, 4+BR-1, ind, val);

	// Display errors and warnings
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_ERR;
}

optimal_value solveOpt(float g, float p, float r, int R, RowVector v)
{
	float q = R/rho;
	optimal_value retval;

	// Objectiv coefficient
	glp_set_obj_coef(lp, 3, -p);
	for(int m=1; m<=BR-1; m++)
	{
		glp_set_obj_coef(lp, m+4, v(m-1));
	}

	// Structural variable bounds
//	glp_set_col_bnds(lp, 1, GLP_UP, 0, fmin(C,1/nuc*g/T)); // Storage charge uc
	glp_set_col_bnds(lp, 1, GLP_UP, 0, C); // Storage charge uc
//	glp_set_col_bnds(lp, 2, GLP_UP, 0, fmin(D,nud*R/rho/T)); // Storage discharge ud
	glp_set_col_bnds(lp, 2, GLP_UP, 0, D); // Storage discharge ud

	// Node balance constraint
	// uc-ud-ggr+rgr = g-r
	glp_set_row_bnds(lp, 1, GLP_FX, g-r, g-r);

	// Value function constraint
	// -T*nuc*uc*rho+T*(1/nud)*ud*rho+sum{r = 0..BR-1}ytr = nul*R
	glp_set_row_bnds(lp, 2, GLP_FX, nul*R, nul*R);

	// Maximum capacity constraint
	// T*nuc*uc-T*(1/nud)*ud <= Qmax-nul*q
	glp_set_row_bnds(lp, 3, GLP_UP, 0, Qmax-nul*q);

	// Minimum capacity constraint
	// T*nuc*uc-T*(1/nud)*ud >= Qmin-nul*q
	glp_set_row_bnds(lp, 4, GLP_LO, Qmin-nul*q, 0);

	// Solve
	glp_simplex(lp, &parm);

	float uc = glp_get_col_prim(lp, 1);
	float ud = glp_get_col_prim(lp, 2);
//	float ggr = glp_get_col_prim(lp, 3);
//	float rgr = glp_get_col_prim(lp, 4);

	retval.F = glp_get_obj_val(lp);
	float Rxerr = floor((nul*q+T*(nuc*uc-(1/nud)*ud))*rho);
	if (Rxerr < 0)
	{
		retval.Rx = 0;
		printf("solveOpt Warning: Negative index: Rxerr=%g\n",Rxerr);
	}
	else if(Rxerr > BR-2)
	{
		retval.Rx = BR-2;
		printf("solveOpt Warning: Too big index: Rxerr=%g\n",Rxerr);
	}
	else retval.Rx = Rxerr;

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
