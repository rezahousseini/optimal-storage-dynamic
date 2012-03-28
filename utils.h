/* ----------------------------------------------------------------------------*
 * vector<int> randi(int min, int max, int length)                             *
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

vector<int> randi(int min, int max, int length) {
	vector<int> sample(length);
	
	for (int k=0; k<length; k++) {
		sample(k) = floor((min+(max+1-min)*rand()/RAND_MAX));
	}
	
	return sample;
}
