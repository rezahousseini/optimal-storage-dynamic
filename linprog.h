struct opt_sol
{
	float F;
	FloatNDArray xc;
	FloatNDArray xd;
	FloatNDArray xh;
	FloatNDArray vhat;
	FloatNDArray phi;
};

/* ----------------------------------------------------------------------------*
 * void initLinProg(void)                                                      *
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
	glp_add_rows(lp, 2*numSfin+2);
	glp_add_cols(lp, 3*numS+numV);
	glp_set_obj_name(lp, "cost");
	glp_set_obj_dir(lp, GLP_MIN);
	int ind[3*numS+numV+1];
	double val[3*numS+numV+1];
	int count;
	char str[80];
	int index;
	
	for (int k=0; k<=3*numS+numV; k++)
	{
		ind[k] = k; // XXX ind and val start at 1 not at 0!!!!
		val[k] = 0;
	}
	
	// Structural variable bounds for u
	for (int m=1; m<=numS; m++)
	{
		sprintf(str, "uc_%i", m);
		glp_set_col_name(lp, m, str); // u charge
		
		sprintf(str, "ud_%i", m);
		glp_set_col_name(lp, numS+m, str); // u discharge
		
		sprintf(str, "uh_%i", m);
		glp_set_col_name(lp, 2*numS+m, str); // u hold
		
		if ((int)set_fin(m-1) == 1)
		{
			glp_set_col_bnds(lp, 2*numS+m, GLP_DB, 
				floor(rho*Qmin(m-1)/T),
				floor(rho*Qmax(m-1)/T)
			);
		}
		else glp_set_col_bnds(lp, 2*numS+m, GLP_FX, 0, 0);
	}
	
	// Structural variable bounds for y
	index = 0;
	for (int m=1; m<=numSfin; m++)
	{
		for (int k=1; k<=(int)numR(m-1)-1; k++)
		{
			sprintf(str, "y_%i_%i", m, k);
			glp_set_col_name(lp, 3*numS+index+k, str);
			glp_set_col_bnds(lp, 3*numS+index+k, GLP_DB, 0, floor(1/T)); // 0 <= yt(r) <= 1
		}
		
		index = index+(int)numR(m-1)-1;
	}
	
	// Node balance constraint
	// -uc+ud+uh = R
	count = 1;
	for (int m=1; m<=numS; m++)
	{
		if ((int)set_fin(m-1) == 1)
		{
			// Reset the val vector to 0
			for (int n=1; n<=3*numS+numV; n++)
			{
				val[n] = 0;
			}
			
			val[m] = -1;// -uc
			val[numS+m] = 1;// ud
			val[2*numS+m] = 1;// uh
			
			sprintf(str, "balance_%i", m);
			glp_set_row_name(lp, count, str);
			glp_set_mat_row(lp, count, 3*numS+numV, ind, val);
			count = count+1;
		}
	}
	
	// Value function constraint
	// -uh+sum{r=1...numR}yt(r) = 0
	count = 1;
	index = 0;
	for (int m=1; m<=numS; m++)
	{
		if ((int)set_fin(m-1) == 1)
		{
			// Reset the val vector to 0
			for (int n=1; n<=3*numS+numV; n++)
			{
				val[n] = 0;
			}
			
			val[m] = 0;// 0*uc
			val[numS+m] = 0;// 0*ud
			val[2*numS+m] = -1;// -uh
			
			for (int k=1; k<=(int)numR(count-1)-1; k++)
			{
				val[3*numS+index+k] = 1;
			}
			index = index+(int)numR(count-1)-1;
			
			sprintf(str, "value_function_%i", count);
			glp_set_row_name(lp, numSfin+count, str);
			glp_set_mat_row(lp, numSfin+count, 3*numS+numV, ind, val);
			glp_set_row_bnds(lp, numSfin+count, GLP_FX, 0, 0);
			
			count = count+1;
		}
	}
	
	// sum uc = ((g-r)+abs(g-r))/2
	// Reset the val vector to 0
	for (int n=1; n<=3*numS+numV; n++)
	{
		val[n] = 0;
	}
	
	for (int m=1; m<=numS; m++)
	{
		val[m] = 1;// uc
	}
	sprintf(str, "input_flow");
	glp_set_row_name(lp, 2*numSfin+1, str);
	glp_set_mat_row(lp, 2*numSfin+1, 3*numS+numV, ind, val);
	
	// sum ud = ((r-g)+abs(r-g))/2
	// Reset the val vector to 0
	for (int n=1; n<=3*numS+numV; n++)
	{
		val[n] = 0;
	}
	
	for (int m=1; m<=numS; m++)
	{
		val[numS+m] = 1; // ud
	}
	sprintf(str, "output_flow");
	glp_set_row_name(lp, 2*numSfin+2, str);
	glp_set_mat_row(lp, 2*numSfin+2, 3*numS+numV, ind, val);
	
	// Display errors and warnings
	glp_init_smcp(&parm_lp);
	parm_lp.msg_lev = GLP_MSG_ERR;
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
	retval.xh = FloatNDArray(dim_vector(numS, 1));
	retval.vhat = FloatNDArray(dim_vector(numSfin, 1));
	retval.phi = FloatNDArray(dim_vector(numV, 1));
	int index = 0;
	int index_v = 0;
	int count = 1;
	int ret;
	
	// Structural variable bounds
	for (int m=1; m<=numS; m++)
	{
		// uc_(k-1)-DeltaCmax <= uc_k <= uc_(k-1)+DeltaCmax
		glp_set_col_bnds(lp, m, GLP_DB,
			floor(fmax(0, xc(m-1)-rho*DeltaCmax(m-1))),
			floor(fmin(rho*C(m-1), xc(m-1)+rho*DeltaCmax(m-1)))
		);
		
		// ud_(k-1)-DeltaDmax <= ud_k <= ud_(k-1)+DeltaDmax
		glp_set_col_bnds(lp, numS+m, GLP_DB,
			floor(fmax(0, xd(m-1)-rho*DeltaDmax(m-1))),
			floor(fmin(rho*D(m-1), xd(m-1)+rho*DeltaDmax(m-1)))
		);
	}
	
	// Objectiv coefficient
	for (int m=1; m<=numS; m++)
	{
		glp_set_obj_coef(lp, m, floor(rho*pc(m-1))); // pc*uc
		glp_set_obj_coef(lp, numS+m, floor(rho*pd(m-1))); // pd*ud
		glp_set_obj_coef(lp, 2*numS+m, 0); // 0*uh
		
		if ((int)set_fin(m-1) == 1)
		{
			for (int k=1; k<=(int)numR(m-1)-1; k++)
			{
				glp_set_obj_coef(lp, 3*numS+index+k, gama*v(index_v+k)); // gama*y*v
			}
			index_v = index_v+(int)numR(m-1);
			index = index+(int)numR(m-1)-1;
			
			// Node balance constraint
			// -uc+ud+uh = R
			glp_set_row_bnds(lp, count, GLP_FX, (float)R(count-1)/T, (float)R(count-1)/T);
			
			count = count+1;
		}
	}
	
	// sum uc = ((g-r)+abs(g-r))/2
	glp_set_row_bnds(lp, 2*numSfin+1, GLP_FX, floor(rho*((g-r)+abs(g-r))/2), floor(rho*((g-r)+abs(g-r))/2));
//	if (g-r < 0) glp_set_row_bnds(lp, 2*numSfin+1, GLP_FX, 0, 0);
//	else glp_set_row_bnds(lp, 2*numSfin+1, GLP_FX, floor(rho*(g-r)), floor(rho*(g-r)));
	
	// sum ud = ((r-g)+abs(r-g))/2
	glp_set_row_bnds(lp, 2*numSfin+2, GLP_FX, floor(rho*((r-g)+abs(r-g))/2), floor(rho*((r-g)+abs(r-g))/2));
//	if (r-g < 0) glp_set_row_bnds(lp, 2*numSfin+2, GLP_FX, 0, 0);
//	else glp_set_row_bnds(lp, 2*numSfin+2, GLP_FX, floor(rho*(r-g)), floor(rho*(r-g)));
	
//	glp_write_lp(lp, NULL, "linearSystem.lp");
	
	// Solve
	ret = glp_simplex(lp, &parm_lp);
	if (ret != 0) printf("No simplex solution. Error %i\n", ret);
	
	retval.F = glp_get_obj_val(lp);
	
	for (int m=1; m<=numS; m++)
	{
		retval.xc(m-1) = glp_get_col_prim(lp, m);
		retval.xd(m-1) = glp_get_col_prim(lp, numS+m);
		retval.xh(m-1) = glp_get_col_prim(lp, 2*numS+m);
		
//		printf("xc_%i=%f\n", m, glp_get_col_prim(lp, m));
//		printf("xd_%i=%f\n", m, glp_get_col_prim(lp, numS+m));
	}
	
	for (int m=1; m<=numSfin; m++)
	{
		retval.vhat(m-1) = glp_get_row_dual(lp, m);
	}
	
	for (int m=1; m<=numV; m++)
	{
		retval.phi(m-1) = glp_get_col_prim(lp, 3*numS+m);
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
