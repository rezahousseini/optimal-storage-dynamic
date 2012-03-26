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
