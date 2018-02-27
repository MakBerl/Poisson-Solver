#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <numeric>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"

#include "cublas_wrapper.h"
#include "misc.h"
#include "preconditioner.h"
#include "solver.h"

/*
 *      test.cu
 *
 *      This file serves as demonstration how to use the given code and runs tests for the different implementations.
 *
 *      @author Simon Schoelly
*/


using namespace std;



int main()
{
  	int device_count;
	cudaError_t err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess || device_count <= 0) {
                cout << "no device found!" << endl;
                abort();
        }


        cublasHandle_t cublas_handle;
        cusparseHandle_t cusparse_handle;
        cublasCreate(&cublas_handle);
        cusparseCreate(&cusparse_handle); 

    	const int    m         = 2048;
	const double alpha     = 0.01f;
        int          max_iter  = 10000;
        double       tolerance = 0.0000001;
        
        double *b, *x;
        cudaMalloc((void **) &b, (m*m)*sizeof(double));
        cudaMalloc((void **) &x, (m*m)*sizeof(double));
        device_memset<double>(b, 1.0, m*m);

        int num_iter = solve_with_conjugate_gradient<double>(cublas_handle, cusparse_handle, m, alpha, b, x, max_iter, tolerance, NULL );
        cout << num_iter << " iterations" << endl;

        SpikeThomasPreconditioner<double> preconditioner(8);
        num_iter = solve_with_conjugate_gradient<double>(cublas_handle, cusparse_handle, m, alpha, b, x, max_iter, tolerance, &preconditioner );
        cout << num_iter << " iterations" << endl;

        ThomasPreconditioner<double> preconditioner2;
        num_iter = solve_with_conjugate_gradient<double>(cublas_handle, cusparse_handle, m, alpha, b, x, max_iter, tolerance, &preconditioner2 );
        cout << num_iter << " iterations" << endl;


        double *b_3d, *x_3d;
        int m_3d = 128;

        cudaMalloc((void **) &b_3d, (m_3d*m_3d*m_3d)*sizeof(double));
        cudaMalloc((void **) &x_3d, (m_3d*m_3d*m_3d)*sizeof(double));
        device_memset<double>(b_3d, 1.0, m*m);

        ThomasPreconditioner3D<double> preconditioner_3d;
        num_iter = solve_with_conjugate_gradient3D<double>(cublas_handle, cusparse_handle, m_3d, alpha, b_3d, x_3d, max_iter, tolerance, &preconditioner_3d);
        cout << num_iter << " iterations" << endl;
         

}
