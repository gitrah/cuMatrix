/*
 * Validation.h
 *
 *  Created on: Feb 21, 2013
 */

#ifndef VALIDATION_H_
#define VALIDATION_H_

#include "Matrix.h"

template<typename T> class Validation {
public:
	static void toValidationSets(Matrix<T>& training,
			Matrix<T>& crossValidation, Matrix<T>& testSet,
			const Matrix<T>& input, T trainingFactor, T crossValidationFactor, vector<uint>& inputIndices, vector<uint>& leftoverIndices) {
		Matrix<T> leftovers;
		input.shuffle(training, leftovers, trainingFactor,inputIndices);
		leftovers.shuffle(crossValidation, testSet,
				crossValidationFactor * input.m / leftovers.m,leftoverIndices);
	}
};

#endif /* VALIDATION_H_ */
