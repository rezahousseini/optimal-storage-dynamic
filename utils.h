/* ----------------------------------------------------------------------------*
 * int32NDArray randi(int min, int max, int length)                            *
 * ----------------------------------------------------------------------------*
 * Generating random integer vector.
 *
 * @param min Minimal number of random vector.
 * @param max Maximal number of random vector.
 * @param length Length of the random integer vector.
 *
 * @return Vector with length <length> random integers between <min> and <max>.
 *
 */

int32NDArray randi(int min, int max, int length)
{
	dim_vector dv;
	dv(0) = length;
	dv(1) = 1;
	int32NDArray sample(dv);
	
	for (int k=0; k<length; k++)
	{
		sample(k) = floor((min+(max+1-min)*rand()/RAND_MAX));
	}
	
	return sample;
}

/* ----------------------------------------------------------------------------*
 * int32NDArray scale(NDArray S, float s)                                      *
 * ----------------------------------------------------------------------------*
 * Scaling an array by a float number.
 *
 * @param S Array to scale.
 * @param s Scale factor.
 *
 * @return Array of integers scaled with factor s.
 *
 */

int32NDArray scale(NDArray S, float s)
{
	dim_vector dv = S.dims();
	int32NDArray S_int(dv);
	
	for (int k=0; k<dv(0); k++)
	{
		for (int m=0; m<dv(1); m++)
		{
			S_int(k,m) = floor(s*S(k,m));
		}
	}
	
	return S_int;
}
