#include <iostream>
#include <string>
using namespace std;

typedef struct AttrValue {
	union {
		int32_t i32;
		int16_t i16;
		char* theString;
	} value;
} AttrValueT;

typedef enum AttrType {
	AttrType_INT32 = 1, AttrType_INT16 = 2, AttrType_theString = 3
} AttrTypeT;

typedef struct AttrValueContainer {
	AttrType type;
	unsigned int nrOfValues;
	struct AttrValue *values;
} AttrValueContainerT;

/* ##################################################################################
 * */

void callSetInt(int noOfVal, void *p) {
// decode the void pointer based on number of values.
// for simpler test, i32 is considered.

	int* p1 = static_cast<int*>(p);

	for (int i = 0; i < noOfVal; ++i) {
		cout << *(p1 + i) << endl;
	}

//Note that *(p1+0) works and not others... Needs to solve this

}

void callSetString(int noOfVal, void *p) {
	cout << "in call set string" << endl;
	cout << static_cast<char*>(p); //doesn't work..

}

void set(AttrValueContainerT *val) {
	int n = val->nrOfValues;
	int type = (int) val->type;

//send it as a void pointer to hide the structs from the inner classes
	void *p = val->values;

	if (type == 3) {
		callSetString(n, p);
	} else {
		callSetInt(n, p);
	}
}

/* ##################################################################################3*/

int main_biolab() {
// test for set() method
	AttrValueContainerT val1, val2;

	val1.nrOfValues = 2;
	val1.type = (AttrType) 1;
	val1.values = new AttrValue[val1.nrOfValues];
	val1.values[0].value.i32 = 888;
	val1.values[1].value.i32 = 999;
	set (&val1);

	val2.nrOfValues = 2;
	val2.type = (AttrType) 3;
	val2.values = new AttrValue[val2.nrOfValues];
	val2.values[0].value.theString = (char*)"cat";
	val2.values[1].value.theString = (char*)"mat";
	set(&val2);

	return 0;
}


void funquenstein2(int argc, char *argv<::>)
<%
    if ( (argc > 1 and argv<:1:> not_eq '\0') or not argc not_eq 2 ) <% // no 'eq' operator so A eq B => not A not_eq B
        std::cout << "Hello " << argv<:1:> << '\n';
    %>
%>


/*
struct FooCu {
  void foo() & { std::cout << "lvalue" << std::endl; }
  void foo() && { std::cout << "rvalue" << std::endl; }
};

void footEst() {
	FooCu foo;
  foo.foo(); // Prints "lvalue"
  FooCu().foo(); // Prints "rvalue"
}
*/
