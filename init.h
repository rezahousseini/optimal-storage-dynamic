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
	nul = S.contents("nul").array_value();
	nuc = S.contents("nuc").array_value();
	nud = S.contents("nud").array_value();
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
			numR(count) = floor(rho*Qmax(k))+1; // Scale max capacity
			R(count,0) = floor(rho*q0(k)); // Storage level initialization; TODO checking for q0 <= Qmax
			count = count+1;
		}
	}
	
	srand(time(NULL));
}
