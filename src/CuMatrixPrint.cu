/*
 * CuMatrixPrint.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"

/////////////////////////////////////////////////////////////////////////
//
// printing
//
/////////////////////////////////////////////////////////////////////////

template<typename T> string CuMatrix<T>::toString() const {

	stringstream ss1,ss2,ss3,sstrout;
	char value[200];

	if(!this->elements) {
		dthrow(noHostBuffer());
	}

	sstrout << "(" ;
	ss1 << m;
	sstrout << ss1.str();
	sstrout << "*" ;
	ss2<< n ;
	sstrout <<  ss2.str();
	sstrout << "*" ;
	ss3 <<  p;
	sstrout << ss3.str();
	sstrout << ")<" << size << "> [" << (colMajor ? "cm]" : "rm]");

	sstrout << " matrix at ";
	sprintf(value, "%p", this);
	sstrout << value;
	sstrout << " h ";
	sprintf(value, "%p", elements);
	sstrout <<  value;
	sstrout << " d ";
	sprintf(value, "%p", d_elements);
	sstrout << value;
	if(d_elements) {
	    cudaPointerAttributes ptrAtts;
	    cherr(cudaPointerGetAttributes(&ptrAtts,d_elements));
		sprintf(value, "<%d>",  ptrAtts.device);
	    sstrout << value;
	}
	sstrout << " {" << b_util::modStr(lastMod) << "}";
	sstrout << "\n";
	bool header = false;
	if (checkDebug(debugVerbose) || (m < getMaxRowsDisplayed() && n < getMaxColsDisplayed())) {
		// print whole matrix
		for (uint i1 = 0; i1 < m; i1++) {
			if(!header) {
				sstrout << "-";
				for (uint j1 = 0; j1 < n; j1++) {
					if(j1 % 10 == 0) {
						sstrout << " " << j1/10;
					}else {
						sstrout << "  ";
					}
					sstrout << " ";
				}
				sstrout << endl;
				header = true;
			}
			sstrout << "[";
			for (uint j1 = 0; j1 < n; j1++) {

				if(sizeof(T) == 4)
					sprintf(value,  (i1 * n + j1) % WARP_SIZE == 0 ? "w%1.4g" : "% 2.4g ", get(i1,j1) );
				else
					sprintf(value, (i1 * n + j1) % WARP_SIZE == 0 ? "w%1.5g" : "% 2.5g", get(i1,j1) );
						//elements[i1 * p + j1]);
				sstrout <<  value;
				if (j1 < n - 1) {
					sstrout <<  " ";
				}
			}
			sstrout << "] ";
			if(i1 % 10 == 0) {
				sstrout  << i1;
			}

			sstrout << "\n";
		}
		if(header) {
			sstrout << "+";
			for (uint j1 = 0; j1 < n; j1++) {
				if(j1 % 10 == 0) {
					sstrout << " " << j1/10;
				}else {
					sstrout << "  ";
				}
				sstrout << " ";
			}
			sstrout << endl;
			header = false;
		}

	} else {
		for (uint i2 = 0; i2 < getMaxRowsDisplayed() + 1 && i2 < m; i2++) {
			if (i2 == getMaxRowsDisplayed()) {
				sstrout <<  ".\n.\n.\n";
				continue;
			}
			for (uint j2 = 0; j2 < getMaxColsDisplayed() + 1 && j2 < n; j2++) {
				if (j2 == getMaxColsDisplayed()) {
					sstrout << "...";
					continue;
				}
				if(sizeof(T) == 4)
					sprintf(value, (i2 * n + j2) % WARP_SIZE == 0 ? "w%1.4g" : "% 2.4g", get(i2,j2));
				else
					sprintf(value, (i2 * n + j2) % WARP_SIZE == 0 ? "w%1.5g" : "% 2.5g", get(i2,j2));
						//elements[i2 * p + j2]);
				sstrout <<  value;
				if (j2 < n - 1) {
					sstrout <<  " ";
				}
			}
			sstrout <<  "\n";
		}
		if (m > getMaxRowsDisplayed()) {
			for (uint i3 =m - getMaxRowsDisplayed(); i3 < m; i3++) {
				if (n > getMaxColsDisplayed()) {
					for (uint j3 = n - getMaxColsDisplayed(); j3 < n; j3++) {
						if (j3 == n - getMaxColsDisplayed()) {
							sstrout << "...";
							continue;
						}
						if(sizeof(T) == 4)
							sprintf(value, (i3 * n + j3) % WARP_SIZE == 0 ? "w%1.4g" : "% 2.4g", get(i3, j3));
						else
							sprintf(value, (i3 * n + j3) % WARP_SIZE == 0 ? "w%1.5g" : "% 2.5g", get(i3,j3));
								//elements[i3 * p + j3]);
						sstrout <<  value;
						if (j3 < n - 1) {
							sstrout << " ";
						}
					}
				} else {
					for (uint j4 = 0; j4 < n; j4++) {
						if(sizeof(T) == 4)
							sprintf(value, (i3 * n + j4) % WARP_SIZE == 0 ? "w%1.4g" : "% 2.4g", get(i3,j4));
						else
							sprintf(value, (i3 * n + j4) % WARP_SIZE == 0 ? "w%1.5g" : "% 2.5g", get(i3,j4));
								//elements[i3 * p + j4]);
						sstrout << value;

						if (j4 < n - 1) {
							sstrout << " ";
						}
					}

				}
				sstrout <<  "\n";
			}
		}
	}
	return sstrout.str();
}

template<typename T> string CuMatrix<T>::pAsRow() {
	poseAsRow();
	string ret = toString();
	unPose();
	return ret;
}

template<typename T> __host__ __device__ void CuMatrix<T>::print(const char* msg) const {

	printf("%s (%d*%d*%d) %d matrix, this %p h %p d %p\n",m,n,p,size,msg, this,elements,d_elements);
	T * elems = NULL;
#ifndef __CUDA_ARCH__
	elems = elements;
#else
	elems = d_elements;
#endif
	if(!elems) {
		return;
	}

	bool header = false;
	if (checkDebug(debugVerbose) || (m < getMaxRowsDisplayed() && n < getMaxColsDisplayed())) {
		for (uint i1 = 0; i1 < m; i1++) {
			if(!header) {
				printf("-");
				for (uint j1 = 0; j1 < n; j1++) {
					if(j1 % 10 == 0) {
						printf(" %d", j1/10);
					}else {
						printf("  ");
					}
					printf(" ");
				}
				printf("\n");
				header = true;
			}
			printf("[");
			for (uint j1 = 0; j1 < n; j1++) {

				if(sizeof(T) == 4)
					printf("%2.3g", elems[i1 * p + j1]);  //get(i1,j1) );
				else
					printf("%2.6g", elems[i1 * p + j1]); // get(i1,j1) );
						//);
				if (j1 < n - 1) {
					printf(" ");
				}
			}
			printf("] ");
			if(i1 % 10 == 0) {
				printf("%d", i1);
			}

			printf("\n");
		}
		if(header) {
			printf("+");
			for (uint j1 = 0; j1 < n; j1++) {
				if(j1 % 10 == 0) {
					printf(" %d",j1/10);
				}else {
					printf("  ");
				}
				printf(" ");
			}
			printf("\n");
			header = false;
		}

	} else {
		for (uint i2 = 0; i2 < getMaxRowsDisplayed() + 1 && i2 < m; i2++) {
			if (i2 == getMaxRowsDisplayed()) {
				printf(".\n.\n.\n");
				continue;
			}
			for (uint j2 = 0; j2 < getMaxColsDisplayed() + 1 && j2 < n; j2++) {
				if (j2 == getMaxColsDisplayed()) {
					printf("...");
					continue;
				}
				if(sizeof(T) == 4)
					printf("%2.3g", elems[i2 * p + j2]); //get(i2,j2));
				else
					printf("%2.6g", elems[i2 * p + j2]); //get(i2,j2));
						//elements[i2 * p + j2]);
				if (j2 < n - 1) {
					printf(" ");
				}
			}
			printf("\n");
		}
		if (m > getMaxRowsDisplayed()) {
			for (uint i3 =m - getMaxRowsDisplayed(); i3 < m; i3++) {
				if (n > getMaxColsDisplayed()) {
					for (uint j3 = n - getMaxColsDisplayed(); j3 < n; j3++) {
						if (j3 == n - getMaxColsDisplayed()) {
							printf("...");
							continue;
						}
						if(sizeof(T) == 4)
							printf("%2.3g", elems[i3 * p + j3]);//get(i3, j3));
						else
							printf("%2.6g", elems[i3 * p + j3]); //get(i3,j3));
								//elements[i3 * p + j3]);
						if (j3 < n - 1) {
							printf(" ");
						}
					}
				} else {
					for (uint j4 = 0; j4 < n; j4++) {
						if(sizeof(T) == 4)
							printf("%2.3g", elems[i3 * p + j4]); // get(i3,j4));
						else
							printf("%2.6g", elems[i3 * p + j4]); //get(i3,j4));
								//elements[i3 * p + j4]);

						if (j4 < n - 1) {
							printf(" ");
						}
					}

				}
				printf("\n");
			}
		} else { //if(m > 10) -> n > 10
			for (uint i5 = 0; i5 < getMaxRowsDisplayed() + 1 && i5 < m; i5++) {

				if (n > getMaxColsDisplayed()) {
					for (uint j5 = n - getMaxColsDisplayed(); j5 < n; j5++) {
						if (j5 == n - getMaxColsDisplayed()) {
							printf("...");
							continue;
						}
						T t = get(i5,j5);

						if(sizeof(T) == 4)
							printf("%2.3g", t);
						else
							printf("%2.6g", t);
						if (j5 < n - 1) {
							printf(" ");
						}
					}
				} else {
					for (uint j4 = 0; j4 < n; j4++) {
						if(sizeof(T) == 4)
							printf("%2.3g", elems[i5 * p + j4]); //get(i5,j4));
						else
							printf("%2.6g", elems[i5 * p + j4]); //get(i5,j4));

						if (j4 < n - 1) {
							printf(" ");
						}
					}
				}

				printf("\n");
			}

		}
	}
}
/*
template void CuMatrix<float>::print(const char*) const;
template void CuMatrix<double>::print(const char*) const;
*/


template<typename T> __host__ __device__ void CuMatrix<T>::printShortString(char const* msg) const {
	const char* pmsg = msg == null ? "" : msg;
#ifndef __CUDA_ARCH__
	cudaPointerAttributes ptrAtts;
	if(d_elements) {
		cherr(cudaPointerGetAttributes(&ptrAtts,d_elements));
		printf("[[ %p %dX%dX%d  (sz %d)  h: %p d: %p <%d>]] %s\n",this, m,n,p,size,elements,d_elements, ptrAtts.device,pmsg);
	} else {
		printf("[[ %p %dX%dX%d  (sz %d)  h: %p d: %p <%d>]] %s\n",this, m,n,p,size,elements,d_elements,pmsg);
	}
#else
	printf("[[ %p %dX%dX%d  (sz %d)  h: %p d: %p ]] %s\n",this, m,n,p,size,elements,d_elements,pmsg);
#endif
}

template<typename T> __host__ __device__ void CuMatrix<T>::printRow(int row) const {
	if(colMajor) {
		setLastError(notImplementedEx);
	}
#ifndef __CUDA_ARCH__
	if(!elements) {
		setLastError(noHostBufferEx);
	}
	if(lastMod == mod_device){
		setLastError(notSyncedHostEx);
	}
#else
	if(!d_elements) {
		setLastError(noDeviceBufferEx);
	}
	if(lastMod == mod_host){
		setLastError(notSynceCUDART_DEVICEEx);
	}
#endif

	ulong idx = row * p;
	for(int c = 0; c < n; c++) {
#ifndef __CUDA_ARCH__
		printf("%5.2f ", elements[idx + c]);
#else
		printf("%5.2f ", d_elements[idx + c]);
#endif
	}
	printf("\n");
}

#include "CuMatrixInster.cu"
