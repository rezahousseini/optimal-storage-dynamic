#include <octave/oct.h>
#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>
#include <octave/ov-struct.h>

DEFUN_DLD(SPARoptimalValueFunctionV0_1, args, nargout, "g,p,r,R,v,rho,T,S")
{
	octave_value_list retval;
	int nargin = args.length();
	if (nargin != 8)
		print_usage();
	else
	{
		float rho = args(0).float_value();
		float g = args(1).float_value();
		float p = args(2).float_value();
		float r = args(3).float_value();
		float R = (float)((args(4).int_value()-1)/rho);
		RowVector v = args(5).row_vector_value();
		float T = args(6).float_value();
		octave_scalar_map S = args(7).scalar_map_value();

		float Qmax = S.contents(0).float_value();
		float q0 = S.contents(1).float_value();
		float C = S.contents(2).float_value();
		float D = S.contents(3).float_value();
		float nul = S.contents(4).float_value();
		float nuc = S.contents(5).float_value();
		float nud = S.contents(6).float_value();

		octave_idx_type BR = v.nelem();

		glp_prob *lp;
		int ret, m;
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
		for(m=1; m<=BR; m++)
		{
			glp_set_col_bnds(lp, m+4, GLP_DB, 0, 1);
		}

		glp_set_obj_coef(lp, 1, 0);
		glp_set_obj_coef(lp, 2, 0);
		glp_set_obj_coef(lp, 3, -p);
		glp_set_obj_coef(lp, 4, 0);
		for(m=1; m<=BR; m++)
		{
			glp_set_obj_coef(lp, m+4, v(0,m));
		}

		ind[1] = 1; val[1] = 1;
		ind[2] = 2; val[2] = -1;
		ind[3] = 3; val[3] = -1;
		ind[4] = 4; val[4] = 1;
		for(m=1; m<=BR; m++)
		{
			ind[m+4] = m+4; val[m+4] = 0;
		}
		glp_set_mat_row(lp, 1, 4+BR, ind, val);
		glp_set_row_bnds(lp, 1, GLP_FX, g-r, g-r);

		ind[1] = 1; val[1] = -1;
		ind[2] = 2; val[2] = 1;
		ind[3] = 3; val[3] = 0;
		ind[4] = 4; val[4] = 0;
		for(m=1; m<=BR; m++)
		{
			ind[m+4] = m+4; val[m+4] = 1;
		}
		glp_set_mat_row(lp, 2, 4+BR, ind, val);
		glp_set_row_bnds(lp, 2, GLP_UP, 0, R);

		ind[1] = 1; val[1] = 1;
		ind[2] = 2; val[2] = -1;
		ind[3] = 3; val[3] = 0;
		ind[4] = 4; val[4] = 0;
		for(m=1; m<=BR; m++)
		{
			ind[m+4] = m+4; val[m+4] = 0;
		}
		glp_set_mat_row(lp, 3, 4+BR, ind, val);
		glp_set_row_bnds(lp, 3, GLP_UP, 0, Qmax-R);

		glp_simplex(lp, NULL);

		float uc = glp_get_col_prim(lp, 1);
		float ud = glp_get_col_prim(lp, 2);
		float ggr = glp_get_col_prim(lp, 3);
		float rgr = glp_get_col_prim(lp, 4);

		retval(0) = octave_value(glp_get_obj_val(lp)-(-p*ggr));
		retval(1) = octave_value((int)((R+uc-ud)*rho)+1);
//		retval(2) = octave_value((int)(glp_get_col_prim(lp, 1)*rho)+1);
//		retval(3) = octave_value((int)(glp_get_col_prim(lp, 2)*rho)+1);
//		retval(4) = octave_value((int)(glp_get_col_prim(lp, 3)*rho)+1);
//		retval(5) = octave_value((int)(glp_get_col_prim(lp, 4)*rho)+1);

		glp_delete_prob(lp);
	}
return retval;
}
