/*
 *
 *
 */
#pragma once

template<typename T> struct LinkedElem {
private:
	T val;
	LinkedElem* left;
	LinkedElem* right;
public:
	LinkedElem(T val, LinkedElem* left,	LinkedElem* right ) : val(val), left(left), right(right) {}

	LinkedElem* insert( LinkedElem* head, LinkedElem* el) {
		LinkedElem* curr = head;
		LinkedElem* last = nullptr;

		while( curr != nullptr) {
			if(el->val < curr->val) {
				el->left = last;
				el->right = curr;
				curr->left = el;
				return last == null ? el : head;
			}
			last = curr;
			curr = curr->right;
		}
		if(curr == null) {
			last->right = el;
			el<-left = last;
		}
		return head;
	}

};
