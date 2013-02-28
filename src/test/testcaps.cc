/*
 * capstest.cc
 *
 *  Created on: Aug 3, 2012
 *      Author: reid
 */

#include <iostream>
#include "../caps.h"
#include "../debug.h"

int
capstest(int argc, char **argv)
{

    ExecCaps ecaps;
    ExecCaps::getExecCaps(ecaps);

    outln("got ecaps " << ecaps.toString());
    KernelCaps kcaps;
    dim3 arrayDim;

    do {
		std::cout << "array len:";
		std::cin >> arrayDim.x;
		arrayDim.y = arrayDim.z = 0;
		outln("got array dim " << b_util::pd3(arrayDim) );
		//std::cout<< "got array len:" << arrayDim.x << std::endl;
		//std::cout<< "got array len:" << arrayDim.x << "," << arrayDim.y << "," << arrayDim.z <<  std::endl;
		if(arrayDim.x > 0) {
			kcaps = KernelCaps::forArray(ecaps, arrayDim);
			outln("got kernel caps block " << b_util::pd3(arrayDim) );
		} else {
			outln("arrayDim.x !> -1");
		}
    }while(arrayDim.x > 0);



    return 0;

}
