#include <helper_cuda.h>
#include  "caps.h"

__device__ uint scanSumAsm(int s) {
	uint tot = 0;
	asm("{\n\t"
			// use braces for local scope
			" .reg.s32 %total, %curr;\n\t"
			" .reg.pred p;\n\t"
			" mov.s32 	%total, 0;\n\t"  // set %total to 0
			" mov.s32 	%curr, %1;\n\t"  // set %curr to arg s
			"sumLuexp: setp.eq.s32 p,%curr,0;\n\t"  // start of loop: set predicate p if %curr is 0
			"@p bra Leavus;\n\t"  // exit if p true
			" add.s32 %total, %total, %curr;\n\t" 	// add %curr to %total
			" sub.s32 %curr, %curr, 1;\n\t" 		// subtract 1 from %curr
			" bra sumLuexp;\n\t"
			"Leavus: \n\t"
			" mov.s32 	%0, %total;\n\t"   // set tot with %total
			"}"
			: "=r"(tot) : "r" (s));

	return tot;
}

__global__ void scanSum(int* d_res, int fin) {
	if(threadIdx.x == 0 && threadIdx.y == 0 && d_res) {
		int isum = scanSumAsm(fin);
		printf("isum for %d is %d\n",fin, isum);
		*d_res = isum;
	}
}

#ifdef CuMatrix_Maxwell_Workaround1
int ompScan(int);
#endif

int scanSumL(int fin) {
	if(fin == 0 || fin == 1) {
		return fin;
	}
	ExecCaps& caps = *ExecCaps::currCaps();
	outln("major " <<caps.deviceProp.major << ", minor " <<  caps.deviceProp.minor);
	if(caps.deviceProp.major == 5 ) {
#ifdef CuMatrix_Maxwell_Workaround1
		return ompScan(fin);
#endif
	}
	int res = 0;
	int* d_res;
	checkCudaErrors(cudaMalloc(&d_res,sizeof(int)));
	scanSum<<<1,1>>>( d_res, fin);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&res,d_res,sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_res));
	return res;
}

