/*
 * AsmTest.cu
 *
 *  Created on: May 18, 2014
 *      Author: reid
 */



__forceinline__ __device__ float2 add(float2 a, float2 b)
{
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

 __device__ float2 sub(float2 a, float2 b)
{
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}
