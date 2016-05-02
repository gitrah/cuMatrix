/*
 * cpp11Features.cc
 *
 *  Created on: May 4, 2014
 *      Author: reid
 */




//Overriding Functions
//catch mistakes when overriding functions
struct B {
	virtual void f(int);
};
struct D1: B {
	void f(int) override;
}; // OK

/*
struct D2: B {
	void f(long) override;
}; // error
*/
struct D3: B {
	void f(long); //hiding intentional?
};

struct S {
	S();
	S(int); // => no implicit default constructor
	virtual S& operator=(S const&) = default;
};
S::S() = default; // non-inline definition

template <typename T1, typename T2>
auto add(T1 first, T2 second) -> decltype(first + second){
	decltype(5) myInt= 5;                                  // C++11
	decltype(3.14) myFloat= 3.14;                          // C++11
	decltype(5) myFloat2= 3.14;                          // C++11

	return first + second;
}


