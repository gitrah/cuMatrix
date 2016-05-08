cuMatrix is a bubbling cauldron of CUDA/c++ experiments, some implementing basic Matrix math
also linear / logistic regression, neural network, anomaly detection functions (so far) 
and functions for reading octave data files

requires Cuda 3.5+ hardware 
	needed to support arbitrary matrix math (including all operators) from device code

Make 
	with kts=1 to effect kernel polymorphism purely by template
	with statFunc=1 for polymorphism by static function pointers
	otherwise, kernels will be built with polymorphism via method pointers 

	(uses a modified ruby script 'ribosome' that does template expansion to switch between these implementations)

	Many matrix operations support extra-resident sizes.

License 

Copyright (c) 2016 Reid Hartenbower 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Todo:

Test performance of passing state
	1.) as argument
	2.) as global mem
	3.) as const mem
	4.) as texture mem
	
