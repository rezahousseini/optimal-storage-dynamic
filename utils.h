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
	
	typedef boost::mt19937 RNGType;
	RNGType rng;
	boost::uniform_int<> randi(min, max-1);
	boost::variate_generator< RNGType, boost::uniform_int<> > dice(rng, randi);
	
	for (int k=0; k<length; k++) {
		sample(k) = dice();
	}
	
	return sample;
}
