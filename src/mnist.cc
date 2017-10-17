#include "mnist.h"
#include <iostream>

using std::ifstream;
using std::ios;

int reverseBits(int v)
{
    uchar c1, c2, c3, c4;
    c1=v & 0xff;
    c2=(v>>8) & 0xff;
    c3=(v>>16) & 0xff;
    c4=(v>>24) & 0xff;
    return((int)c1<<24) | ((int)c2<<16) | ((int)c3<<8) | c4;
}

template <typename T> CuMatrix<T> readMnistImages(const char* path ) {
    ifstream file (path,ios::binary);
    if (file.is_open())   {
        int magic=0;
        int samples=0;
        int h=0;
        int w=0;
        file.read((char*)&magic,sizeof(magic));
        magic= reverseBits(magic);
        file.read((char*)&samples,sizeof(samples));
        samples= reverseBits(samples);
        file.read((char*)&h,sizeof(h));
        h= reverseBits(h);
        file.read((char*)&w,sizeof(w));
        w= reverseBits(w);

        outln("samples " << samples);
        outln("h " << h);
        outln("w " << w);
    	CuMatrix<T> res = CuMatrix<T>::zeros(samples, h * w);

        unsigned char temp=0;
        for(int i=0;i<samples;++i)  {
        	//if(i > 0 && i % 100 == 0) { outln("read " << i << " samples");}
            for(int r=0;r<h;++r) {
                for(int c=0;c<w;++c) {
                    file.read((char*)&temp,sizeof(temp));
                    res.elements[ i * res.p + h*r +c ] = (T) temp;
                }
            }
        }
        res.invalidateDevice();
        return res;
    }
    return CuMatrix<T>::ZeroMatrix;
}
template CuMatrix<float> readMnistImages<float>(char const*);
template CuMatrix<double> readMnistImages<double>(char const*);
template CuMatrix<unsigned long> readMnistImages<unsigned long>(char const*);

template <typename T> CuMatrix<T> readMnistLables(const char* path) {
    ifstream file (path,ios::binary);
    if (file.is_open())
    {
        int magic=0;
        int samples=0;
        file.read((char*)&magic,sizeof(magic));
        magic= reverseBits(magic);
        file.read((char*)&samples,sizeof(samples));
        samples= reverseBits(samples);

        outln("samples " << samples);
    	CuMatrix<T> res = CuMatrix<T>::zeros(samples, 1);

		unsigned char temp=0;
        for(int i=0;i<samples;++i)  {
			//if(i > 0 && i % 100 == 0) { outln("read " << i << " samples");}
			file.read((char*)&temp,sizeof(temp));
			res.elements[ i ] = (T) temp;
        }
        res.invalidateDevice();
        return res;
    }
    return CuMatrix<T>::ZeroMatrix;
}
template CuMatrix<float> readMnistLables<float>(char const*);
template CuMatrix<double> readMnistLables<double>(char const*);
template CuMatrix<unsigned long> readMnistLables<unsigned long>(char const*);
