#ifndef STCSTEPSIZE_H_
#define STCSTEPSIZE_H_

#include "Stepsize.h"

class STCStepsize : public Stepsize {
	public:
		STCStepsize(float alpha0, float _beta, float _a, float _b);
		void update(void);
	protected:
		float beta;
		float a;
		float b;
};

#endif /* STCSTEPSIZE_H_ */
