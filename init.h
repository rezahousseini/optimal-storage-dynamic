/* ----------------------------------------------------------------------------*
 * int32NDArray init(octave_scalar_map S)                                              *
 * ----------------------------------------------------------------------------*
 * Initiation of the algorithm.
 *
 * @param S Structure with storage parameters.
 *
 * @return R
 *
 */ 

int32NDArray init(octave_scalar_map S)
{
	// Storage parameters
	Qmax = S.contents("Qmax").array_value();
	Qmin = S.contents("Qmin").array_value();
	q0 = S.contents("q0").array_value();
	C = S.contents("C").array_value();
	D = S.contents("D").array_value();
	etal = S.contents("etal").array_value();
	etac = S.contents("etac").array_value();
	etad = S.contents("etad").array_value();
	DeltaCmax = S.contents("DeltaCmax").array_value();
	DeltaDmax = S.contents("DeltaDmax").array_value();
	
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
	numR = int32NDArray(dim_vector(numSfin, 1));
	
	int32NDArray R(dim_vector(numSfin, numN), 0);
	
	count = 0;
	for (int k=0; k<numS; k++)
	{
		if ((int)set_fin(k) == 1)
		{
			numR(count) = floor(rho*Qmax(k))+1; // Scale max capacity
			R(count, 0) = floor(rho*q0(k)); // Storage level initialization; TODO checking for q0 <= Qmax
			count = count+1;
		}
	}
	
	srand(time(NULL));
	
	return R;
}
