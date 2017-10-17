/*
 * Validation.h
 *
 *  Created on: Feb 21, 2013
 */

#pragma once

#include "CuMatrix.h"

template<typename T> class Validation {
public:
	static void toValidationSets(CuMatrix<T>& training,
			CuMatrix<T>& crossValidation, CuMatrix<T>& testSet,
			const CuMatrix<T>& input, T trainingFactor, T crossValidationFactor, vector<int>& inputIndices, vector<int>& leftoverIndices) {
		CuMatrix<T> leftovers;
		input.shuffle(&training, &leftovers, trainingFactor,inputIndices);
		leftovers.shuffle(&crossValidation, &testSet,
				crossValidationFactor * input.m / leftovers.m,leftoverIndices);
	}
};
