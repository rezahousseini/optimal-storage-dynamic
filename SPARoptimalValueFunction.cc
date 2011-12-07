#include <octave/oct.h>
#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>

DEFUN_DLD(SPARoptimalValueFunctionV0_0, args, nargout, "filename")
{
	octave_value_list retval;
	int nargin = args.length();
	if (nargin != 1)
		print_usage();
	else
	{
		charMatrix name = args(0).char_matrix_value();

		charMatrix suffixDat = ".dat";
		charMatrix suffixMod = ".mod";
		charMatrix suffixClp = ".clp";
		charMatrix suffixSol = ".sol";
		octave_idx_type len = name.nelem();

		glp_prob *lp;
		glp_tran *tran;
		int ret;

		lp = glp_create_prob();
		tran = glp_mpl_alloc_wksp();

		name.insert(suffixMod, 0, len);
		ret = glp_mpl_read_model(tran, name.row_as_string(0).c_str(), 1);
		if (ret != 0)
		{
			fprintf(stderr, "Error on translating model\n");
			goto skip;
		}

		name.insert(suffixDat, 0, len);
		ret = glp_mpl_read_data(tran, name.row_as_string(0).c_str());
		if (ret != 0)
		{
			fprintf(stderr, "Error on translating data\n");
			goto skip;
		}

		ret = glp_mpl_generate(tran, NULL);
		if (ret != 0)
		{
			fprintf(stderr, "Error on generating model\n");
			goto skip;
		}

		glp_mpl_build_prob(tran, lp);

		glp_simplex(lp, NULL);

		name.insert(suffixClp, 0, len);
		ret = glp_write_lp(lp, NULL, name.row_as_string(0).c_str());
		if (ret != 0)
		{
			fprintf(stderr, "Error on writing problem\n");
			goto skip;
		}

		skip: 

		retval(0) = octave_value(glp_get_obj_val(lp));
		retval(1) = octave_value(glp_get_col_prim(lp, 1));
		retval(2) = octave_value(glp_get_col_prim(lp, 2));
		retval(3) = octave_value(glp_get_col_prim(lp, 3));

		glp_mpl_free_wksp(tran);
		glp_delete_prob(lp);
	}
return retval;
}
