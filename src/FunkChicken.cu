/*
 * FunkChicken.cc
 *
 *  Created on: Jul 1, 2014
 *      Author: reid
 */
#include <iostream>

template<typename TYPE>
class Target5 {
public:
	Target5(const TYPE &value) :
			value(value) {
	}
	TYPE value;

	template<typename T>
	void OneParam(T a) {
		std::cout <<"Target5::OneParam("<<value<<","<<a<<")\n";

		typedef void (Target5<TYPE>::*MethodTypeToCall)(T);
		// Here, the compiler picks the right overload
		MethodTypeToCall toCall = &Target5<TYPE>::Private;
		// In this case, the compiler does not let us write the following line (parse error):
		//MethodTypeToCall toCall = &Target5<TYPE>::Private<t;;
		(this->*toCall)(a);
	}

	template<typename T1, typename T2>
	void TwoParam(T1 a, T2 b) {
		std::cout << "Target5::TwoParam(" << value << "," << a << "," << b
				<< ")\n";

		typedef void (Target5<TYPE>::*MethodTypeToCall)(T1, T2);
		MethodTypeToCall toCall = &Target5<TYPE>::Private; // compiler picks the right overload
		// you can't add the method's template parameters to the end of that line
		(this->*toCall)(a, b);
	}

private:

	template<typename T>
	void Private(T a) {
		std::cout << "Target5::Private(" << value << "," << a << ")\n";
	}
	template<typename T1, typename T2>
	void Private(T1 a, T2 b) {
		std::cout << "Target5::Private(" << value << "," << a << "," << b
				<< ")\n";
	}
};

void HoldingAPointerToTemplateMemberInTemplateClass() {
	Target5<char> target('c');

	void (Target5<char>::*oneParam)(int) = &Target5<char>::OneParam;
	(target.*oneParam)(7);
	void (Target5<char>::*twoParam)(float, int) = &Target5<char>::TwoParam;
	(target.*twoParam)(7.5, 7);
}
