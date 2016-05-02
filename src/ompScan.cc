/*
 * ompScan.cc
 *
 *  Created on: Dec 7, 2014
 *      Author: reid
 */

#include "util.h"

int ompScan(int n) {
	int tot = 0;
/*
	uint np2_n = b_util::nextPowerOf2(n);
	uint* ns = new uint[np2_n];
	for(uint i =0;i < np2_n; i++) {
		ns[i]= i < n? i+1 : 0;
	}

	delete[] ns;
*/
	while(n > 0) {
		tot += n;
		n--;
	}
	return tot;
}
