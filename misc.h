#ifndef MISC_H
#define MISC_H

#include "cublas_v2.h"
#include "cusparse_v2.h"

/*
 *      misc.h
 *
 *      This file contains miscellaneous helper functions.
 *
 *      @author Simon Schoelly
 */

int divide_and_round_up(int const n, int const d) {
        if (n % d == 0) {
                return n/d;
        }
        return (n/d + 1);
}

template<class FT>
void print_device_array(FT const * const array, size_t const num_elements, char const * const array_symbol) {
        std::cout << array_symbol << " (" << num_elements << ") elements:";
	FT *host_array = new FT[num_elements];
	cudaMemcpy(host_array, array, num_elements * sizeof(FT), cudaMemcpyDeviceToHost);
	for (int i = 0; i <num_elements; ++i) {
                std::cout << " " << host_array[i];
	}
        std::cout << std::endl;
	delete[] host_array;
}

template<class T>
__global__ void device_memset_kernel(T * const devPtr, T const value, size_t const count, size_t offset) {
        size_t const tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= count) {
                return;
        }
        devPtr[tid + offset] = value;
}

template<class T> 
void device_memset(T * const devPtr, T const value, size_t const count, size_t offset=0) {
        if (count <= 0) {
                return;
        }
        device_memset_kernel<T><<<divide_and_round_up(count, 32), 32>>>(devPtr, value, count, offset);
}

#endif
