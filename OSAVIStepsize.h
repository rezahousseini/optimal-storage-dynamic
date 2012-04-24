#ifndef OSAVISTEPSIZE_H_
#define OSAVISTEPSIZE_H_

#include "Stepsize.h"

class OSAVIStepsize : public Stepsize {
	public:
		OSAVIStepsize(float alpha0, float c0, float _nu, float sigma0, float _gamma);
		void update(float cost);
	protected:
		float c;
		float nu;
		float sigma;
		float gamma;
		float lambda;
		float delta;
	private:
		void update(void);
};

#endif /* OSAVISTEPSIZE_H_ */

