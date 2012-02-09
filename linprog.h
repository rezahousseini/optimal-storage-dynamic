struct opt_sol
{
	float F;
	float C;
	FloatNDArray xc;
	FloatNDArray xd;
	FloatNDArray vhat;
	FloatNDArray phi;
};

/* ----------------------------------------------------------------------------*
 * void initLinProg(void)                                                          *
 * ----------------------------------------------------------------------------*
 * Initiat the linear programming problem.
 *
 * @param void
 *
 * @return void
 *
 */

void initLinProg(void)
{
	lp = glp_create_prob();
	int numV = (int)numR.sum(0).elem(0)-numSfin;
	glp_add_rows(lp, 1+2*numSfin);//+numS);
	glp_add_cols(lp, 2*numS+numV);
	glp_set_obj_name(lp, "profit");
	glp_set_obj_dir(lp, GLP_MIN);
	int ind[2*numS+numV+1];
	double val[2*numS+numV+1];
	int count;
	char str[80];
	int index;
	
	for (int k=1; k<=2*numS+numV; k++)
	{
		ind[k] = k; // XXX ind and val start at 1 not at 0!!!!
	}
	
	// Structural variable bounds
	index = 0;
	for (int m=0; m<numSfin; m++)
	{
		sprintf(str, "uc_%i", m+1);
		glp_set_col_name(lp, m+1, str);
		
		sprintf(str, "ud_%i", m+1);
		glp_set_col_name(lp, numS+m+1, str);
		
		for (int k=1; k<=(int)numR(m)-1; k++)
		{
			sprintf(str, "y_%i_%i", m+1, k);
			glp_set_col_name(lp, 2*numS+index+k, str);
			glp_set_col_bnds(lp, 2*numS+index+k, GLP_DB, 0, 1); // 0 <= ytr <= 1
		}
		index = index+(int)numR(m)-1;
	}
	
	// Node balance constraint
	// uc-ud = g-r
	glp_set_row_name(lp, 1, "balance");
	for (int m=1; m<=numS; m++)
	{
		val[m] = 1;//nuc(m-1); // uc
		val[numS+m] = -1;//-1/nud(m-1); // -ud
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
		// XXX Which variables are integer?
//		glp_set_col_kind(lp, m, GLP_IV);
//		glp_set_col_kind(lp, numS+m, GLP_IV);
		
		if ((int)set_fin(m-1) == 1)
		{
			
			
			// Reset the val vector to 0
			for (int n=1; n<=2*numS+numV; n++)
			{
				val[n] = 0;
			}
			
			val[m] = -1;//-nuc(m-1); // -uc
			val[numS+m] = 1;//1/nud(m-1); // ud
			
			for (int k=1; k<=(int)numR(count)-1; k++)
			{
				val[2*numS+index+k] = 1;
			}
			index = index+(int)numR(count)-1;
			
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
			
			val[m] = 1;//nuc(m-1); // uc
			val[numS+m] = -1;//-1/nud(m-1); // -ud
			
			sprintf(str, "capacity_bound_%i", count);
			glp_set_row_name(lp, 1+numSfin+count, str);
			glp_set_mat_row(lp, 1+numSfin+count, 2*numS+numV, ind, val);
			count = count+1;
		}
	}
	
	// Display errors and warnings
	glp_init_smcp(&parm_lp);
	parm_lp.msg_lev = GLP_MSG_ERR;
//	parm_lp.tol_bnd = 1e-4;
}

/* --------------------------------------------------------------------------- *
 * opt_sol solveLinProg(float g, float r, FloatNDArray pc, FloatNDArray pd,    *
 *  int32NDArray R, FloatNDArray v, FloatNDArray xc, FloatNDArray xd)          *
 * --------------------------------------------------------------------------- *
 * Solving the linear programming problem.
 *
 * @param g  Generated energy (node input to satisfie). 1 x 1
 * @param r Requested energy (node output to satisfie). 1 x 1
 * @param pc Price vector for charge vaiables of all resources. numS x 1
 * @param pd Price vector for discharge vaiables of all resources. numS x 1
 * @param R Resource level vector for all finite capacities. numSfin x 1
 * @param v Value function approximation for all level steps. sum(numR) x 1
 * @param xc
 * @param xd
 *
 * @return Optimal solution argument and value.
 *
 */

opt_sol solveLinProg(float g, float r, FloatNDArray pc, FloatNDArray pd,
	int32NDArray R, FloatNDArray v, FloatNDArray xc, FloatNDArray xd)
{
	opt_sol retval;
	int numV = (int)numR.sum(0).elem(0)-numSfin;
	retval.xc = FloatNDArray(dim_vector(numS, 1));
	retval.xd = FloatNDArray(dim_vector(numS, 1));
	retval.vhat = FloatNDArray(dim_vector(numSfin, 1));
	retval.phi = FloatNDArray(dim_vector(numV, 1));
	int index = 0;
	int index_v = 0;
	int count = 1;
	int ret;
	
	// Structural variable bounds
	for (int m=1; m<=numS; m++)
	{
//		// uc_(k-1)-DeltaCmax <= uc_k <= uc_(k-1)+DeltaCmax
//		float xc_max = floor(T*nuc(m-1)*fmin(rho*C(m-1), (xc(m-1)+rho*DeltaCmax(m-1))));
//		float xc_min = floor(T*nuc(m-1)*fmax(0, (xc(m-1)-rho*DeltaCmax(m-1))));
//		
//		if (xc_max == xc_min)
//		{
//			glp_set_col_bnds(lp, m, GLP_FX, xc_min, xc_max);
//		}
//		else
//		{
//			glp_set_col_bnds(lp, m, GLP_DB, xc_min, xc_max);
//		}
//		
//		// ud_(k-1)-DeltaDmax <= ud_k <= ud_(k-1)+DeltaDmax
//		float xd_max = floor(T/nud(m-1)*fmin(rho*D(m-1), (xd(m-1)+rho*DeltaDmax(m-1))));
//		float xd_min = floor(T/nud(m-1)*fmax(0, (xd(m-1)-rho*DeltaDmax(m-1))));
//		
//		if (xd_max == xd_min)
//		{
//			glp_set_col_bnds(lp, numS+m, GLP_FX, xd_min, xd_max);
//		}
//		else
//		{
//			glp_set_col_bnds(lp, numS+m, GLP_DB, xd_min, xd_max);
//		}
		
		// uc_(k-1)-DeltaCmax <= uc_k <= uc_(k-1)+DeltaCmax
		glp_set_col_bnds(lp, m, GLP_DB, 0, floor(rho*C(m-1)));
		
		// ud_(k-1)-DeltaDmax <= ud_k <= ud_(k-1)+DeltaDmax
		glp_set_col_bnds(lp, numS+m, GLP_DB, 0, floor(rho*D(m-1)));
	}
	
	// Objectiv coefficient
	for (int m=1; m<=numS; m++)
	{
		glp_set_obj_coef(lp, m, floor(rho*pc(m-1))); // -pc*uc
		glp_set_obj_coef(lp, numS+m, floor(rho*pd(m-1))); // -pd*ud
		
		if ((int)set_fin(m-1) == 1)
		{
			for (int k=1; k<=(int)numR(m-1)-1; k++)
			{
//				printf("gama*v(index_v+k)=%f\n", gama*v(index_v+k));
				glp_set_obj_coef(lp, 2*numS+index+k, gama*v(index_v+k)); // gama*y*v
			}
			index_v = index_v+(int)numR(m-1);
			index = index+(int)numR(m-1)-1;
			
			// Value function constraint
			// -uc+ud+sum{r = 0..numR-1}ytr = R
//			glp_set_row_bnds(lp, 1+count, GLP_FX,
//				floor(nul(m-1)*(float)R(count-1)),
//				floor(nul(m-1)*(float)R(count-1))
//			);
			glp_set_row_bnds(lp, 1+count, GLP_FX,
				floor((float)R(count-1)),
				floor((float)R(count-1))
			);
			
			// Minimum and Maximum capacity constraint
			// Qmin-R <= uc-ud <= Qmax-R
//			glp_set_row_bnds(lp, 1+numSfin+count, GLP_DB,
//				floor(rho*Qmin(m-1)-nul(m-1)*(float)R(count-1)),
//				floor(rho*Qmax(m-1)-nul(m-1)*(float)R(count-1))
//			);
			glp_set_row_bnds(lp, 1+numSfin+count, GLP_DB,
				floor((rho*Qmin(m-1)-(float)R(count-1))),
				floor((rho*Qmax(m-1)-(float)R(count-1)))
			);
			
			count = count+1;
		}
	}
	
	// Node balance constraint
	// uc-ud = g-r
	glp_set_row_bnds(lp, 1, GLP_FX, floor(rho*(g-r)), floor(rho*(g-r)));
	
//	glp_write_lp(lp, NULL, "linearSystem.lp");
//	glp_write_mps(lp, GLP_MPS_FILE, NULL, "linearSystem.mps");
	
	// Solve
	ret = glp_simplex(lp, &parm_lp);
	if (ret != 0)
	{
		printf("No simplex solution. Error %i\n", ret);
		for (int m=1; m<=numS; m++)
		{
			printf("xc_%i=%f\n", m, xc(m-1));
			printf("xd_%i=%f\n", m, xd(m-1));
		}
	}
	
	retval.F = glp_get_obj_val(lp);
	float C = 0;
	
	for (int m=1; m<=numS; m++)
	{
		retval.xc(m-1) = glp_get_col_prim(lp, m);
		retval.xd(m-1) = glp_get_col_prim(lp, numS+m);
		
		if (glp_get_col_prim(lp, m) < 0)
		{
			printf("xc_%i=%f\n", m, glp_get_col_prim(lp, m));
		}
		
		if (glp_get_col_prim(lp, numS+m) < 0)
		{
			printf("xd_%i=%f\n", m, glp_get_col_prim(lp, numS+m));
		}
		
		C = C-glp_get_col_prim(lp, m)/rho*pc(m-1)-glp_get_col_prim(lp, numS+m)/rho*pd(m-1);
	}
//	printf("\n");
	retval.C = C;
	
	for (int m=1; m<=numSfin; m++)
	{
		retval.vhat(m-1) = glp_get_row_dual(lp, 1+m);
	}
	
	for (int m=1; m<=numV; m++)
	{
		retval.phi(m-1) = glp_get_col_prim(lp, 2*numS+m);
	}
	
	return retval;
}

/* ----------------------------------------------------------------------------*
 * void deleteLinProg(void)                                                        *
 * ----------------------------------------------------------------------------*
 * Deleting the linear programming problem.
 *
 * @param void
 *
 * @return void
 *
 */

void deleteLinProg(void)
{
	glp_delete_prob(lp);
}
