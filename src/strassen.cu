#include<iostream>
#include<vector>
#include "test/tests.h"

using std::vector;
using std::cout;

__device__ void sum(int* A, int* B, int* C, int tam) {
	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			C[i * tam + j] = A[i * tam + j] + B[i * tam + j];
		}
	}
}

__device__ void subtract(int* A, int* B, int* C, int tam) {
	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			C[i * tam + j] = A[i * tam + j] - B[i * tam + j];
		}
	}
}

__device__ void strassenR(int* A, int* B, int* C, int tam) {
//if (tam <= leafsize) {
//    ikjalgorithm(A, B, C, tam);
//    return;
//}
	if (tam == 1) {
		C[0] = A[0] * B[0];
		return;
	}

// other cases are treated here:
	else {
		int newTam = tam / 2;

		int* a11 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* a12 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* a21 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* a22 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* b11 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* b12 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* b21 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* b22 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* c11 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* c12 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* c21 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* c22 = (int*) malloc(sizeof(int) * newTam * newTam);

		int* p1 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* p2 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* p3 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* p4 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* p5 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* p6 = (int*) malloc(sizeof(int) * newTam * newTam);
		int* p7 = (int*) malloc(sizeof(int) * newTam * newTam);

		int* aResult = (int*) malloc(sizeof(int) * newTam * newTam);
		int* bResult = (int*) malloc(sizeof(int) * newTam * newTam);

		int i, j;

		//dividing the matrices in 4 sub-matrices:
		for (i = 0; i < newTam; i++) {
			for (j = 0; j < newTam; j++) {
				a11[i * newTam + j] = A[i * tam + j];
				a12[i * newTam + j] = A[i * tam + j + newTam];
				a21[i * newTam + j] = A[(i + newTam) * tam + j];
				a22[i * newTam + j] = A[(i + newTam) * tam + j + newTam];

				b11[i * newTam + j] = B[i * tam + j];
				b12[i * newTam + j] = B[i * tam + j + newTam];
				b21[i * newTam + j] = B[(i + newTam) * tam + j];
				b22[i * newTam + j] = B[(i + newTam) * tam + j + newTam];
			}
		}

		// Calculating p1 to p7:

		sum(a11, a22, aResult, newTam); // a11 + a22
		sum(b11, b22, bResult, newTam); // b11 + b22
		strassenR(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

		sum(a21, a22, aResult, newTam); // a21 + a22
		strassenR(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

		subtract(b12, b22, bResult, newTam); // b12 - b22
		strassenR(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

		subtract(b21, b11, bResult, newTam); // b21 - b11
		strassenR(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

		sum(a11, a12, aResult, newTam); // a11 + a12
		strassenR(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)

		subtract(a21, a11, aResult, newTam); // a21 - a11
		sum(b11, b12, bResult, newTam); // b11 + b12
		strassenR(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

		subtract(a12, a22, aResult, newTam); // a12 - a22
		sum(b21, b22, bResult, newTam); // b21 + b22
		strassenR(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

		// calculating c21, c21, c11 e c22:

		sum(p3, p5, c12, newTam); // c12 = p3 + p5
		sum(p2, p4, c21, newTam); // c21 = p2 + p4

		sum(p1, p4, aResult, newTam); // p1 + p4
		sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
		subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7

		sum(p1, p3, aResult, newTam); // p1 + p3
		sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
		subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6

		// Grouping the results obtained in a single matrix:
		for (i = 0; i < newTam; i++) {
			for (j = 0; j < newTam; j++) {
				C[i * tam + j] = c11[i * newTam + j];
				C[i * tam + j + newTam] = c12[i * newTam + j];
				C[(i + newTam) * tam + j] = c21[i * newTam + j];
				C[(i + newTam) * tam + j + newTam] = c22[i * newTam + j];
			}
		}

		/*delete a11;   delete a12; delete a21; delete a22;
		 delete b11; delete b12; delete b21; delete b22;
		 delete c11; delete c12; delete c21; delete c22;
		 delete p1;  delete p2;  delete p3;  delete p4;
		 delete p5;  delete p6;  delete p7;
		 delete aResult; delete bResult;*/
	}
}

//__device__ WorkStack stack;

__global__ void kernel(int* A, int* B, int* C, int N) {

//__shared__ int data1[17 * 2];
//int* a11 = (int*)malloc(sizeof(int)*4*N);

	strassenR(A, B, C, N);

}

unsigned int nextPowerOfTwo(float n) {
	return powf(2, int(ceil(::log2(n))));
}

void printMatrix(vector<vector<int> > matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (j != 0) {
				cout << "\t";
			}
			cout << matrix[i][j];
		}
		cout << endl;
	}
}
void printMatrix(int* matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (j != 0) {
				cout << "\t";
			}
			cout << matrix[i * n + j];
		}
		cout << endl;
	}
}

template int testStrassen<float>::operator()(int argc, const char **argv) const;
template int testStrassen<double>::operator()(int argc, const char **argv) const;
template <typename T> int testStrassen<T>::operator()(int argc, const char **argv) const {

	/*   int A[]={1,3,4,
	 3,1,2};

	 int B[]={1,3,4,0,
	 2,1,0,1,
	 0,2,2,1};
	 int rowsA = 2, colsA = 3;
	 int rowsB = 3, colsB = 4;
	 */
	int A[] = { 1, 3, 3, 1 };
	int B[] = { 1, 3, 2, 1 };
	int rowsA = 2, colsA = 2;
	int rowsB = 2, colsB = 2;

	if (colsA != rowsB) {
		cout << "error: cosA is not equal to rowsB!" << endl;
		return 0;
	}

	int N1 = max(max(rowsA, colsA), colsB);
	int N = nextPowerOfTwo((float) N1);
	cout << "N is : " << N << endl;

//vector<int> inner (n);
//vector< vector<int> > A(n, inner), B(n, inner), C(n, inner);

	vector<int> inner(N);
	vector<vector<int> > APrep(N, inner), BPrep(N, inner), CPrep(N, inner);

// resize A and B, filling with zeros
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < colsA; j++) {
			APrep[i][j] = A[i * colsA + j];
		}
	}
	cout << "APrep : " << endl;
	printMatrix(APrep, N);
	for (int i = 0; i < rowsB; i++) {
		for (int j = 0; j < colsB; j++) {
			BPrep[i][j] = B[i * colsB + j];
		}
	}
	cout << "BPrep : " << endl;
	printMatrix(BPrep, N);

//strassenR(APrep, BPrep, CPrep, N);  // recursive strassen
//cout<<"CPrep (result): "<<endl;
//printMatrix(CPrep,N);

// matrix to array
	int A_new[N * N], B_new[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A_new[i * N + j] = APrep[i][j];
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			B_new[i * N + j] = BPrep[i][j];
		}
	}

	/*int newTam = 2;
	 int tam = 4;
	 int* a11 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* a12 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* a21 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* a22 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* b11 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* b12 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* b21 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* b22 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* c11 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* c12 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* c21 = (int*)malloc(sizeof(int)*newTam*newTam);
	 int* c22 = (int*)malloc(sizeof(int)*newTam*newTam);
	 for (int i = 0; i < newTam; i++) {
	 for (int j = 0; j < newTam; j++) {
	 a11[i*newTam+j] = A_new[i*tam+j];
	 a12[i*newTam+j] = A_new[i*tam+j + newTam];
	 a21[i*newTam+j] = A_new[(i + newTam)*tam+j];
	 a22[i*newTam+j] = A_new[(i + newTam)*tam+j + newTam];

	 b11[i*newTam+j] = B_new[i*tam+j];
	 b12[i*newTam+j] = B_new[i*tam+j + newTam];
	 b21[i*newTam+j] = B_new[(i + newTam)*tam+j];
	 b22[i*newTam+j] = B_new[(i + newTam)*tam+j + newTam];
	 }
	 }
	 cout<<"a11 : "<<endl;   printMatrix(a11,newTam);
	 cout<<"a12 : "<<endl;   printMatrix(a12,newTam);
	 cout<<"a21 : "<<endl;   printMatrix(a21,newTam);
	 cout<<"a22 : "<<endl;   printMatrix(a22,newTam);
	 cout<<"b11 : "<<endl;   printMatrix(b11,newTam);
	 cout<<"b12 : "<<endl;   printMatrix(b12,newTam);
	 cout<<"b21 : "<<endl;   printMatrix(b21,newTam);
	 cout<<"b22 : "<<endl;   printMatrix(b22,newTam);*/

	int *A_new_d;
	cudaMalloc((void**) &A_new_d, sizeof(int) * N * N);
	cudaMemcpy(A_new_d, A_new, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	int *B_new_d;
	cudaMalloc((void**) &B_new_d, sizeof(int) * N * N);
	cudaMemcpy(B_new_d, B_new, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	int *C_new_d;
	cudaMalloc((void**) &C_new_d, sizeof(int) * N * N);

	cudaThreadSetLimit(cudaLimitStackSize, 128 * 1024 * 1024); //cudaLimitMallocHeapSize
	kernel<<<1,1>>>(A_new_d, B_new_d, C_new_d, N);
//cudaDeviceSynchronize();

	int* C_out = (int*) malloc(sizeof(int) * N * N);
	cudaMemcpy(C_out, C_new_d, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	cout << "C_out : " << endl;
	printMatrix(C_out, N);

	cudaFree(A_new_d);
	cudaFree(B_new_d);
	cudaFree(C_new_d);
	delete C_out;

	return 0;
}
