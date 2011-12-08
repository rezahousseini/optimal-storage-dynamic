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
optimal_value solveOpt(int g, int p, int r, int R, RowVector v);
void deleteOpt(void);
intNDArray<int> randi(int start, int end, int number);

DEFUN_DLD(SPARoptimal1Storage, args, nargout, "W,S,Iter,rho,T")
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
		rho = args(3).int_value();
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

		intNDArray<int> R(dv_R,0); // Pre-decision asset level
		intNDArray<int> Rx(dv_R,0); // Post-decision asset level
		BR = floor(rho*Qmax);
		Matrix v(BW*N,BR);
		Matrix NV(BW*N,BR); // Number of visits to the corresponding state
		int R0 = floor(rho*q0); // Storage level initialization
		optimal_value ret, ret1, ret2;
		float vhatlo, vhatup;
		Matrix z(1,2);
		float alpha;
		intNDArray<int> sample;
		RowVector q(N);

		v.fill(0);
		NV.fill(1);
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
				ret = solveOpt(floor(rho*g(sample(k),k)),floor(rho*p(sample(k),k)),floor(rho*r(sample(k),k)),R(k),v.row(sample(k)*N+k));
				Rx(k) = ret.Rx;

				// Count number of visits
				NV(sample(k)*N+k,Rx(k)) = NV(sample(k)*N+k,Rx(k))+1;

				if(k < N-1)
				{
					// Observe sample slopes
					ret1 = solveOpt(floor(rho*g(sample(k+1),k+1)),floor(rho*p(sample(k+1),k+1)),floor(rho*r(sample(k+1),k+1)),Rx(k),v.row(sample(k+1)*N+k+1));
					if(Rx(k) == 0) vhatlo = ret1.F;
					else
					{
						ret2 = solveOpt(floor(rho*g(sample(k+1),k+1)),floor(rho*p(sample(k+1),k+1)),floor(rho*r(sample(k+1),k+1)),Rx(k)-1,v.row(sample(k+1)*N+k+1));
						vhatlo = ret1.F-ret2.F;
					}

					ret2 = solveOpt(floor(rho*g(sample(k+1),k+1)),floor(rho*p(sample(k+1),k+1)),floor(rho*r(sample(k+1),k+1)),Rx(k),v.row(sample(k+1)*N+k+1));
					if(Rx(k) == BR-1) vhatup = 0;
					else
					{
						ret1 = solveOpt(floor(rho*g(sample(k+1),k+1)),floor(rho*p(sample(k+1),k+1)),floor(rho*r(sample(k+1),k+1)),Rx(k)+1,v.row(sample(k+1)*N+k+1));
						vhatup = ret1.F-ret2.F;
					}

					// Update slopes
					// Calculate alpha
					alpha = 1/NV(sample(k)*N+k,Rx(k));

					// Calculate z
					z(0) = (1-alpha)*v(sample(k)*N+k,Rx(k))+alpha*vhatlo;
					if(Rx(k)+1 < BR)
					{
						z(1) = (1-alpha)*v(sample(k)*N+k,Rx(k)+1)+alpha*vhatup;
					}
					else
					{
						z(1) = vhatup;
					}

					// Projection operation
					v(sample(k)*N+k,Rx(k)) = z(0);
					if(Rx(k)+1 < BR)
					{
						v(sample(k)*N+k,Rx(k)+1) = z(1);
					}

					for(int level=0; level<BR; level++)
					{
						if(level < Rx(k) and v(sample(k)*N+k,level) <= z(0))
						{
							v(sample(k)*N+k,level) = z(0);
						}
						else if(level > (Rx(k)+1) and v(sample(k)*N+k,level) >= z(1))
						{
							v(sample(k)*N+k,level) = z(1);
						}
					} // endfor level
				} // endif
			} // endfor k
		} // endfor iter
		for(int k=0; k<N; k++)
		{
			q(k) = R(k)/rho;
		}

		retval(0) = octave_value(Rx+1);
		retval(1) = octave_value(v);

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
	glp_add_rows(lp, 3);
	glp_add_cols(lp, 4+BR-1);
	glp_set_obj_dir(lp, GLP_MAX);

	// Structural variable bounds
	glp_set_col_bnds(lp, 3, GLP_LO, 0, 0);
	glp_set_col_bnds(lp, 4, GLP_LO, 0, 0);
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
	ind[1] = 1; val[1] = -T*nuc;
	ind[2] = 2; val[2] = T*(1/nud);
	ind[3] = 3; val[3] = 0;
	ind[4] = 4; val[4] = 0;
	for(int m=1; m<=BR-1; m++)
	{
		ind[m+4] = m+4; val[m+4] = 1;
	}
	glp_set_mat_row(lp, 2, 4+BR-1, ind, val);

	// Minimum and Maximum capacity constraint
	// Qmin-nul*q <= T*nuc*uc-T*(1/nud)*ud <= Qmax-nul*q
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

optimal_value solveOpt(int g, int p, int r, int R, RowVector v)
{
	optimal_value retval;

	// Objectiv coefficient
	glp_set_obj_coef(lp, 3, -p);
	for(int m=1; m<=BR-1; m++)
	{
		glp_set_obj_coef(lp, m+4, v(m-1));
	}

	// Structural variable bounds
	glp_set_col_bnds(lp, 1, GLP_DB, 0, floor(C*rho)); // Storage charge uc
	glp_set_col_bnds(lp, 2, GLP_DB, 0, floor(D*rho)); // Storage discharge ud

	// Node balance constraint
	// uc-ud-ggr+rgr = g-r
	glp_set_row_bnds(lp, 1, GLP_FX, g-r, g-r);

	// Value function constraint
	// -T*nuc*uc*rho+T*(1/nud)*ud*rho+sum{r = 0..BR-1}ytr = nul*R
	glp_set_row_bnds(lp, 2, GLP_FX, nul*R, nul*R);

	// Minimum and Maximum capacity constraint
	// Qmin-nul*q <= T*nuc*uc-T*(1/nud)*ud <= Qmax-nul*q
	glp_set_row_bnds(lp, 3, GLP_DB, floor(rho*Qmin)-nul*R, BR-nul*R);

	// Solve
	glp_simplex(lp, &parm);

	float uc = glp_get_col_prim(lp, 1);
	float ud = glp_get_col_prim(lp, 2);
	float ggr = glp_get_col_prim(lp, 3);
	float rgr = glp_get_col_prim(lp, 4);
	RowVector y(BR);
	for(int m=1; m<=BR-1; m++)
	{
		y(m) = glp_get_col_prim(lp, m+4);
	}

	retval.F = glp_get_obj_val(lp);
	int Rxerr = floor(nul*R+T*(nuc*uc-(1/nud)*ud));
	if (Rxerr < 0)
	{
		retval.Rx = 0;
		printf("solveOpt Warning: Negative index: Rxerr=%i\n",Rxerr);
	}
	else if(Rxerr > BR-1)
	{
		retval.Rx = BR-1;
		printf("solveOpt Warning: Too big index: Rxerr=%i\n",Rxerr);
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
