// Array A[] has the items to sort; array B[] is a work array.
typedef unsigned int uint;

template<typename T> class TopDownMergeSort<T> {

public:

	TopDownMergeSort(T a[], T b[], uint n) {
		TopDownSplitMerge(a, 0, n, b);
	}

// iBegin is inclusive; iEnd is exclusive (A[iEnd] is not in the set).
	void TopDownSplitMerge(T a[], uint iBegin, uint iEnd, T b[]) {
		if (iEnd - iBegin < 2)                       // if run size == 1
			return;                                 //   consider it sorted
		// recursively split runs into two halves until run size == 1,
		// then merge them and return back up the call chain
		iMiddle = (iEnd + iBegin) / 2;              // iMiddle = mid point
		TopDownSplitMerge(a, iBegin, iMiddle, b);  // split / merge left  half
		TopDownSplitMerge(a, iMiddle, iEnd, b);  // split / merge right half
		TopDownMerge(a, iBegin, iMiddle, iEnd, b);  // merge the two half runs
		CopyArray(b, iBegin, iEnd, a);         // copy the merged runs back to A
	}

//  Left half is A[iBegin:iMiddle-1].
// Right half is A[iMiddle:iEnd-1   ].
	void TopDownMerge(T a[], uint iBegin, uint iMiddle, uint iEnd, T b[]) {
		uint i = iBegin, j = iMiddle;

		// While there are elements in the left or right runs...
		for (k = iBegin; k < iEnd; k++) {
			// If left run head exists and is <= existing right run head.
			if (i < iMiddle && (j >= iEnd || a[i] <= a[j])) {
				b[k] = a[i];
				i = i + 1;
			} else {
				b[k] = a[j];
				j = j + 1;
			}
		}
	}

	void CopyArray(T b[], uint iBegin, uint iEnd, T a[]) {
		for (k = iBegin; k < iEnd; k++)
			a[k] = b[k];
	}
};
