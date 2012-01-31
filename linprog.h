struct opt_sol
{
	float F;
	float C;
	FloatNDArray xc;
	FloatNDArray xd;
	FloatNDArray vhat;
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
	glp_set_obj_dir(lp, GLP_MAX);
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
			
			val[m] = -nuc(m-1); // -uc
			val[numS+m] = 1/nud(m-1); // ud
			
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
			
			val[m] = nuc(m-1); // uc
			val[numS+m] = -1/nud(m-1); // -ud
			
			sprintf(str, "capacity_bound_%i", count);
			glp_set_row_name(lp, 1+numSfin+count, str);
			glp_set_mat_row(lp, 1+numSfin+count, 2*numS+numV, ind, val);
			count = count+1;
		}
	}
	
//	// Velocity of power change constraint
//	// -DeltaDmax <= (ud_k-uc_k)-(ud_(k-1)-uc_(k-1)) <= DeltaCmax
//	for (int m=1; m<=numS; m++)
//	{
//		// Reset the val vector to 0
//		for (int n=1; n<=2*numS+numV; n++)
//		{
//			val[n] = 0;
//		}
//		
//		val[m] = nuc(m-1); // uc
//		val[numS+m] = 1/nud(m-1); // ud
//		
//		sprintf(str, "u_change_bound_%i", m);
//		glp_set_row_name(lp, 1+2*numSfin+m, str);
//		glp_set_mat_row(lp, 1+2*numSfin+m, 2*numS+numV, ind, val);
//	}
	
	// Display errors and warnings
	glp_init_smcp(&parm_lp);
	parm_lp.msg_lev = GLP_MSG_ERR;
//	parm_lp.tol_bnd = 1e-4;
	
//	glp_init_iocp(&parm_mip);
//	parm_mip.msg_lev = GLP_MSG_ERR;
//	parm_mip.presolve = GLP_ON;
//	parm_mip.gmi_cuts = GLP_ON;
//	parm_mip.tol_int = 1e-5;
//	parm_mip.mip_gap = 0.001;
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
	retval.xc = FloatNDArray(dim_vector(numS, 1));
	retval.xd = FloatNDArray(dim_vector(numS, 1));
	retval.vhat = FloatNDArray(dim_vector(numSfin, 1));
	int index = 0;
	int index_v = 0;
	int count = 1;
	int ret;
	
	// Structural variable bounds
	for (int m=1; m<=numS; m++)
	{
		// uc_(k-1)-DeltaCmax <= uc_k <= uc_(k-1)+DeltaCmax
		glp_set_col_bnds(lp, m, GLP_DB,
			fmax(0, floor(T*nuc(m-1)*(xc(m-1)-rho*DeltaCmax(m-1)))),
			fmin(
				floor(T*nuc(m-1)*rho*C(m-1)),
				floor(T*nuc(m-1)*(xc(m-1)+rho*DeltaCmax(m-1)))
			)
		);
		
		// ud_(k-1)-DeltaDmax <= ud_k <= ud_(k-1)+DeltaDmax
		glp_set_col_bnds(lp, numS+m, GLP_DB,
			fmax(0, floor(T/nud(m-1)*(xd(m-1)-rho*DeltaDmax(m-1)))),
			fmin(
				floor(T/nud(m-1)*rho*D(m-1)),
				floor(T/nud(m-1)*(xd(m-1)+rho*DeltaDmax(m-1)))
			)
		);
	}
	
	// Objectiv coefficient
	for (int m=1; m<=numS; m++)
	{
		glp_set_obj_coef(lp, m, -pc(m-1)); // -pc*uc
		glp_set_obj_coef(lp, numS+m, -pd(m-1)); // -pd*ud
		
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
			glp_set_row_bnds(lp, 1+count, GLP_FX, floor((float)R(count-1)*nul(m-1)), floor((float)R(count-1)*nul(m-1)));
			
			// Minimum and Maximum capacity constraint
			// Qmin-R <= uc-ud <= Qmax-R
			glp_set_row_bnds(lp, 1+numSfin+count, GLP_DB, floor((rho*Qmin(m-1)-nul(m-1)*(float)R(count-1))),
				floor((rho*Qmax(m-1)-nul(m-1)*(float)R(count-1))));
			
			count = count+1;
		}
		
//		// Velocity of power change constraint
//		// -DeltaDmax <= (ud_k-uc_k)-(ud_(k-1)-uc_(k-1)) <= DeltaCmax
//		glp_set_row_bnds(lp, 1+2*numSfin+m, GLP_DB, -rho*DeltaDmax(m-1)+(xd(m-1)-xc(m-1)), rho*DeltaCmax(m-1)+(xd(m-1)-xc(m-1)));
	}
	
	// Node balance constraint
	// uc-ud = g-r
	glp_set_row_bnds(lp, 1, GLP_FX, floor(rho*(g-r)), floor(rho*(g-r)));
	
//	glp_write_lp(lp, NULL, "linearSystem.lp");
//	glp_write_mps(lp, GLP_MPS_FILE, NULL, "linearSystem.mps");
	
	// Solve
	glp_simplex(lp, &parm_lp);
	
//	for (int m=1; m<=numS; m++)
//	{
//		printf("xc1=%f\n", glp_get_col_prim(lp, m));
//		printf("xd1=%f\n", glp_get_col_prim(lp, numS+m));
//	}
	
//	ret = glp_intopt(lp, &parm_mip);
//	if (ret != 0)
//	{
//		printf("No intopt solution. Error %i\n", ret);
//	}
	
	retval.F = glp_get_obj_val(lp);
	float C = 0;
//	retval.F = glp_mip_obj_val(lp);
	
	for (int m=1; m<=numS; m++)
	{
//		retval.xc(m-1) = glp_mip_col_val(lp, m);
//		retval.xd(m-1) = glp_mip_col_val(lp, numS+m);
		
		
		retval.xc(m-1) = glp_get_col_prim(lp, m);
		retval.xd(m-1) = glp_get_col_prim(lp, numS+m);
		C = C-glp_get_col_prim(lp, m)/rho*pc(m-1)-glp_get_col_prim(lp, numS+m)/rho*pd(m-1);
//		printf("xc2=%f\n", glp_mip_col_val(lp, m));
//		printf("xd2=%f\n", glp_mip_col_val(lp, numS+m));
	}
	retval.C = C;
	
	for (int m=1; m<=numSfin; m++)
	{
		retval.vhat(m-1) = glp_get_row_dual(lp, 1+m);
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
