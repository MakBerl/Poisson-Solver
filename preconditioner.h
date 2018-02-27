#include "misc.h"
#include "cublas_wrapper.h"

/*
 *      preconditioner.h
 *
 *      This file contains different preconditioner implementations. They are all implementations of the abstract
 *      type Preconditioner. It is the users responsibility to check wether a preconditioner solves the 2D or 
 *      the 3D problem.
 *
 *      @author Simon Schoelly
 *
 */

/*
 *      Abstract preconditioner class. Provides two methods that have to be implemented:
 *      init: initialzes the preconditioner before the first run
 *      run: solves M\b
 *
 *      @param FT Field Type - Either float or double
 *      
 */
template<class FT>
class Preconditioner {
public:
        virtual void init(int const m, FT const alpha, cublasHandle_t cublas_handle, cusparseHandle_t cusparse_handle) = 0;
        virtual void run(FT const * const b, FT * const x) = 0;
};


/*
 *      Kernel that performs the thomas algorithm. Used for ThomasPreconditioner and SpikeThomasPreconditioner.
 *
 *      @param FT Field Type - Either float or double
 *      @param m grid size i.e. M is of size mxm
 *      @param m >= block_size >= 1 the size of a block that is inverted. For the ThomasPreconditioner this is of size m
 *      @param num_blocks the number of blocks that we invert
 *      @param alpha > 0.
 *      @param beta = sqrt(alpha)
 *      @param c_prime coefficients for the thomas algorithm that where precualculated
 *      @param b != NULL input vector of size m*num_blocks*block_size
 *      @param x != NULL output vector of size m*num_blocks*block_size
 *
 *      @return M\X  
 *      
 */
template<class FT>
__global__ void thomas_kernel(int const m, int block_size, int num_blocks, FT const alpha, FT const beta, FT const * const __restrict__ dev_c_prime,  FT const * const __restrict__ b, FT * const __restrict__ x) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m*num_blocks) {
	  return;
	}

        int start = (tid / m) * block_size * m + (tid % m);

        FT work_buffer_reg = b[start] * beta / (FT(2) + alpha);

        x[start] = work_buffer_reg;
        for (int i = 1; i < block_size; ++i) { 
                int j = start + i*m;
                work_buffer_reg  = (b[j] * beta + work_buffer_reg) / (FT(2) + alpha + dev_c_prime[i-1]);
                x[j] = work_buffer_reg;
        }
        
        FT x_reg = x[start + (block_size-1)*m];
        x[start + (block_size-1)*m] = x_reg;
        for (int i = block_size-2; i >= 0; --i) {
                int j = start + i*m;
                x_reg = x[j] - dev_c_prime[i] * x_reg;
                x[j] = x_reg;
        }
}

/*
 *      Preconditioner for the 2D problem. Uses the thomas algorithm to invert M.
 *
 *      @param FT Field Type - Either float or double
 *      
 */
template<class FT>
class ThomasPreconditioner : public Preconditioner<FT> {
private:
        FT *c_prime;
        int m;
        FT alpha, beta;
        cublasHandle_t cublas_handle;
        FT *b_trans;

        int thomas_kernel_block_size;
public:
        virtual void init(int const m, FT const alpha, cublasHandle_t cublas_handle, cusparseHandle_t cusparse_handle) {
                this->m = m;
                this->alpha = alpha;
                beta = sqrt(alpha);
                this-> cublas_handle = cublas_handle;

                FT *host_c_prime = new FT[m];

                host_c_prime[0] = FT(-1) / (alpha + FT(2));
                for (int i = 1; i < m; ++i) {
                        host_c_prime[i] = FT(-1) / (host_c_prime[i-1] + FT(2) + alpha);
                }

                cudaMalloc((void **) &c_prime, m*sizeof(FT));
                cudaMemcpy(c_prime, host_c_prime, m*sizeof(FT), cudaMemcpyHostToDevice);

                delete[] host_c_prime;

                cudaMalloc((void **) &b_trans, m*m*sizeof(FT));

                int minGridSize;
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &thomas_kernel_block_size, thomas_kernel<FT>, 0, m); 
                thomas_kernel_block_size /= 8;
        }
        
        virtual void run(FT const * const b, FT * const x) {
                FT block_count =  divide_and_round_up(m, thomas_kernel_block_size); 
                FT threads_per_block = thomas_kernel_block_size;

                cublas_transpose(cublas_handle, m, b, b_trans);
                FT *y_trans = x; 
                thomas_kernel<FT><<<block_count, threads_per_block>>>(m, m, 1, alpha, beta, c_prime, b_trans, y_trans);
                FT *y = b_trans; 
                cublas_transpose(cublas_handle, m, y_trans, y);

                thomas_kernel<FT><<<block_count, threads_per_block>>>(m, m, 1, alpha, beta, c_prime, y, x);
        }

        ~ThomasPreconditioner() {
                cudaFree(b_trans);
                cudaFree(c_prime);
        }

};

/*
 *      Kernel that performs the thomas algorithm for the 3D problem. Used for ThomasPreconditioner3
 *
 *      @param FT Field Type - Either float or double
 *      @param m grid size i.e. M is of size mxm
 *      @param n number of vectors that we invert simultanously. Usually has value m*m
 *      @param alpha > 0.
 *      @param alpha_23 = alpha^(2/3)
 *      @param c_prime coefficients for the thomas algorithm that where precualculated
 *      @param b != NULL input vector of size m*n
 *      @param x != NULL output vector of size m*n
 *
 *      @return M\X  
 *      
 */

template<class FT>
__global__ void thomas_kernel3D(int const m, int n, FT const alpha, FT const alpha_23, FT const * const __restrict__ dev_c_prime,  FT const * const __restrict__ b, FT * const __restrict__ x) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) {
	  return;
	}

        int start = tid;


        FT work_buffer_reg = b[start] * alpha_23 / (FT(2) + alpha);

        x[start] = work_buffer_reg;
        for (int i = 1; i < m; ++i) { 
                int j = start + i*n;
                work_buffer_reg  = (b[j] * alpha_23 + work_buffer_reg) / (FT(2) + alpha + dev_c_prime[i-1]);
                x[j] = work_buffer_reg;
        }
        
        FT x_reg = x[start + (m-1)*n];
        x[start + (m-1)*n] = x_reg;
        for (int i = m-2; i >= 0; --i) {
                int j = start + i*n;
                x_reg = x[j] - dev_c_prime[i] * x_reg;
                x[j] = x_reg;
        }
}

/*
 *      Preconditioner for the 3D problem. Uses the thomas algorithm to invert M.
 *
 *      @param FT Field Type - Either float or double
 *      
 */
template<class FT>
class ThomasPreconditioner3D : public Preconditioner<FT> {
private:
        FT *c_prime;
        int m;
        cublasHandle_t cublas_handle;
        FT *b_trans;
        FT alpha, alpha_23;

        int thomas_kernel_block_size;
public:
        virtual void init(int const m, FT const alpha, cublasHandle_t cublas_handle, cusparseHandle_t cusparse_handle) {
                this->m = m;
                this->alpha = alpha;
                this-> cublas_handle = cublas_handle;

                FT *host_c_prime = new FT[m];

                alpha_23 = pow(alpha, FT(2)/FT(3));

                host_c_prime[0] = FT(-1) / (alpha + FT(2));
                for (int i = 1; i < m; ++i) {
                        host_c_prime[i] = FT(-1) / (host_c_prime[i-1] + FT(2) + alpha);
                }

                cudaMalloc((void **) &c_prime, m*sizeof(FT));
                cudaMemcpy(c_prime, host_c_prime, m*sizeof(FT), cudaMemcpyHostToDevice);

                delete[] host_c_prime;

                cudaMalloc((void **) &b_trans, m*m*m*sizeof(FT));

                int minGridSize;
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &thomas_kernel_block_size, thomas_kernel3D<FT>, 0, m*m); 
        }
        
        virtual void run(FT const * const b, FT * const x) {
                FT block_count =  divide_and_round_up(m*m, thomas_kernel_block_size); 
                FT threads_per_block = thomas_kernel_block_size;

                int n = m*m*m;


                FT *y = x; 
                thomas_kernel3D<FT><<<block_count, threads_per_block>>>(m,  m*m, alpha, alpha_23, c_prime, b, y);

                FT *y_trans = b_trans; 
                cublas_transpose2(cublas_handle, m*m, m, y, y_trans);

                FT *z_trans = y;
                thomas_kernel3D<FT><<<block_count, threads_per_block>>>(m,  m*m, alpha, alpha_23, c_prime, y_trans, z_trans);

                FT *z_trans2 = y_trans;
                cublas_transpose2(cublas_handle, m*m, m, z_trans, z_trans2);
                FT *x_trans2 = z_trans;

                thomas_kernel3D<FT><<<block_count, threads_per_block>>>(m,  m*m, alpha, alpha_23, c_prime, z_trans2, x_trans2);
                FT *x_trans = z_trans2;
                cublas_transpose2(cublas_handle, m, m*m, x_trans2, x_trans);

                cublas_transpose2(cublas_handle, m, m*m, x_trans, x);
        }

        ~ThomasPreconditioner3D() {
                cudaFree(c_prime);
                cudaFree(b_trans);
        }

};



/*
 *      Kernel that extract coefficients at the block borders for SpikeThomasPreconditioner
 *
 *      @param FT Field Type - Either float or double
 *      @param m grid size i.e. M is of size mxm
 *      @param block_size size of a block
 *      @param nLu = 2*num_blocks + 2
 *      @param b != NULL input vector of size m*m
 *      @param x != NULL output vector of size m*nLU
 *      
 */
template<class FT>
__global__ void ST_block_borders_kernel(int const m, int const block_size, int const nLU, FT const * const __restrict__ b, FT * const __restrict__ b2) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= m * nLU  ) {
                return;
        }
        int col2 = tid / m;
        int row2 = tid % m;
        int block_num = (col2+1) / 2;
        int col = block_size * block_num + (((col2 +1) % 2) * (block_size - 1) ); 
        b2[col2*m + row2] = b[col*m + row2];
}


/*
 *      Kernel that performs the backsubstition after we alrady solved the problem at the borders of the blocks. 
 *      Used for SpikeThomasPreconditioner
 *
 *      @param FT Field Type - Either float or double
 *      @param m grid size i.e. M is of size mxm
 *      @param block_size size of a block
 *      @param left_spike the left spike of size block_size
 *      @param right_spike the right spike of size block_size
 *      @param nLu = 2*num_blocks + 2
 *      @param b != NULL input vector of size m*m
 *      @param x2 != NULL input vector of size m*nLU that contains the solutions at the borders of the blocks
 *      @param x != NULL output vector of size m*m
 *      
 */
template<class FT>
__global__ void ST_backsubstitution_kernel(int const m, int const block_size, int nLU, FT const * const __restrict__ left_spike, FT const * const __restrict__ right_spike, FT const * const __restrict__ b, FT const * const __restrict__ x2, FT * const __restrict__ x) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= m*m) {
                return;
        }
        int col = tid / m;
        int row = tid % m;
        int num_blocks = m / block_size;
        int block_num = col / block_size;
        int spike_index = col % block_size;
        int col2 = 2 * block_num * m + row;
        if (block_num == 0) {
                x[tid] = b[tid] - right_spike[spike_index] * x2[col2 + m];

        } else if (block_num == num_blocks - 1)  {
                x[tid] = b[tid] - left_spike[spike_index] * x2[col2 - 2*m];
        } else {
                x[tid] = b[tid] - left_spike[spike_index] * x2[col2 - 2*m] - right_spike[spike_index] * x2[col2 + m];
        }
}

/*
 *      Kernel that performs the forwards substitution to solve the spike matrix for SpikeThomasPreconditioner
 *      Used for SpikeThomasPreconditioner. It is assumed that the diagonal l0 is 1 everywhere.
 *
 *      @param FT Field Type - Either float or double
 *      @param m grid size i.e. M is of size mxm
 *      @param nLu = 2*num_blocks + 2
 *      @param l1 contains the l1 digonal of L in the LU factorisation of the spike matrix.
 *      @param l2 contains the l2 digonal of L in the LU factorisation of the spike matrix.
 *      @param X != NULL input vector of size m*nLU 
 *      @param Y != NULL output vector of size m*nLU
 *      
 */
template<class FT>
__global__ void ST_lu_forwards_kernel(int const m, int const nLU, FT const * const __restrict__ l1, FT const * const __restrict__ l2, FT const * const __restrict__ X, FT * const __restrict__ Y) {
	        int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid > m) {
                        return;
                }
                FT y1, y2;

                y2 = X[tid];
                y1 = X[tid+m] - y2 * l1[0];
                Y[tid] = y2;
                Y[tid+m] = y1;
                FT y0;
                for (int i = 2; i < nLU; ++i) {
                        y0 = (X[tid + i*m] - y1 * l1[i-1]) - y2 * l2[i-2];
                        Y[tid + i*m] = y0;
                        y2 = y1;
                        y1 = y0;
                }
}


/*
 *      Kernel that performs the backwards substitution to solve the spike matrix for SpikeThomasPreconditioner
 *      Used for SpikeThomasPreconditioner. 
 *
 *      @param FT Field Type - Either float or double
 *      @param m grid size i.e. M is of size mxm
 *      @param nLu = 2*num_blocks + 2
 *      @param u0 contains the u0 digonal of U in the LU factorisation of the spike matrix. Can not contains entries of value 0.
 *      @param u1 contains the u1 digonal of U in the LU factorisation of the spike matrix.
 *      @param u2 contains the u2 digonal of U in the LU factorisation of the spike matrix.
 *      @param X != NULL input vector of size m*nLU 
 *      @param Y != NULL output vector of size m*nLU
 *      
 */
template<class FT>
__global__ void ST_lu_backwards_kernel(int const m, int const nLU, FT const * const __restrict__ u0, FT const * const __restrict__ u1, FT const * const __restrict__ u2, FT const * const __restrict__ X, FT * const __restrict__ Y) {
	        int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid > m) {
                        return;
                }
                FT y0, y1, y2;
                y2 =  X[tid + (nLU-1)*m] / u0[nLU-1];
                y1 = (X[tid + (nLU-2)*m] - y2 * u1[nLU-2]) / u0[nLU-2];


                Y[tid + (nLU-1)*m] = y2;
                Y[tid + (nLU-2)*m] = y1;
                for (int i = nLU-3; i >= 0; --i) {
                        y0 = (X[tid + i*m] - y1 * u1[i] - y2 * u2[i]) / u0[i];
                        Y[tid + i*m] = y0;
                        y2 = y1;
                        y1 = y0;
                }
}

/*
 *      Preconditioner for the 2D problem. This uses the Spike algorithm to distribute the data over more threads
 *      and then uses the thomas algorithm to invert M.
 *
 *      @param FT Field Type - Either float or double
 *      @param num_blocks >= 2 The number of additional subdivisions we create. Should be a divisor of m
 *      
 */ 
template<class FT>
class SpikeThomasPreconditioner : public Preconditioner<FT> {
private:
        int num_blocks;
        int block_size;
        int m;
        cublasHandle_t cublas_handle;

        int nLU;

        FT *b2;
        FT *Ux2;
        FT *x2;
        FT *c, *c_trans, *b_trans;

        FT *c_prime;
        FT *y;

        FT alpha, beta;

        FT *l1, *l2;
        FT *u0, *u1, *u2;

        FT *left_spike, *right_spike;
        FT *A_right_spike;
        FT *A_left_spike;

        int SpikeThomas_thomas_kernel_trans2_block_size;
        int forwards_substitution_kernel3_trans_threads_per_block;
        int backwards_substitution_kernel3_trans_threads_per_block;

        int thomas_kernel_block_size;
        int ST_block_borders_kernel_block_size;
        int ST_backsubstitution_kernel_block_size;
        int ST_lu_forwards_kernel_block_size;
        int ST_lu_backwards_kernel_block_size;
public:
        SpikeThomasPreconditioner(int num_blocks) {
                this->num_blocks = num_blocks;
        }

        virtual void init(int const m, FT const alpha, cublasHandle_t cublas_handle, cusparseHandle_t cusparse_handle) {
                this->m = m;
                this->cublas_handle = cublas_handle;
                this->alpha = alpha;
                beta = sqrt(alpha);

                while (m % num_blocks != 0) {
                        --num_blocks;
                }


                block_size = m / num_blocks;


                cudaMalloc((void **) &left_spike, block_size*sizeof(FT));
                cudaMalloc((void **) &A_left_spike, block_size*sizeof(FT));
                cudaMalloc((void **) &right_spike, block_size*sizeof(FT));
                cudaMalloc((void **) &A_right_spike, block_size*sizeof(FT));

                cudaMalloc((void **) &c_prime, block_size*sizeof(FT));
                FT *host_c_prime = new FT[block_size];

                host_c_prime[0] = FT(-1) / (alpha + FT(2));
                for (int i = 1; i < block_size; ++i) {
                        host_c_prime[i] = FT(-1) / (host_c_prime[i-1] + FT(2) + alpha);
                }

                cudaMemcpy(c_prime, host_c_prime, block_size*sizeof(FT), cudaMemcpyHostToDevice);

                device_memset<FT>(A_right_spike, FT(0), block_size-1);
                device_memset<FT>(A_right_spike, FT(-1)/beta, 1, block_size-1);


                thomas_kernel<FT><<<1, 1>>>(1, block_size, 1, alpha, beta, c_prime, A_right_spike, right_spike);


                device_memset<FT>(A_left_spike, FT(0), block_size-1, 1);
                device_memset<FT>(A_left_spike, FT(-1)/beta, 1); 

                thomas_kernel<FT><<<1, 1>>>(1, block_size, 1, alpha, beta, c_prime, A_left_spike, left_spike);

                nLU = 2*num_blocks - 2;

                FT v1, v2, w1, w2; 
                cudaMemcpy(&v1, &right_spike[0], sizeof(FT), cudaMemcpyDeviceToHost);
                cudaMemcpy(&v2, &right_spike[block_size-1], sizeof(FT), cudaMemcpyDeviceToHost);
                cudaMemcpy(&w1, &left_spike[0], sizeof(FT), cudaMemcpyDeviceToHost);
                cudaMemcpy(&w2, &left_spike[block_size-1], sizeof(FT), cudaMemcpyDeviceToHost);

                FT *host_u0 = new FT[nLU];
                FT *host_u1 = new FT[nLU-1];
                FT *host_u2 = new FT[nLU-2];
                FT *host_l1 = new FT[nLU-1];
                FT *host_l2 = new FT[nLU-2];

                host_u0[0] = FT(1);
                host_u1[0] = v2;
                host_u2[0] = FT(0);


                for (int i = 0; i < num_blocks - 2; ++i) {
                        FT factor1 = -w1;        
                        FT factor2 = -w2;        
                        host_u0[2*i+1] = FT(1) + factor1 * host_u1[2*i];
                        host_u1[2*i+1] = FT(0);
                        host_u2[2*i+1] = v1;
                        host_u0[2*i+2] = FT(1);
                        FT factor3 = -factor2 * host_u1[2*i]   / host_u0[2*i+1];
                        host_u1[2*i+2] = v2 + factor3 * v1;
                        if (2*i+2 < nLU-2) {
                                host_u2[2*i+2] = FT(0);
                        }

                        host_l1[2*i] = -factor1;
                        host_l1[2*i+1] = -factor3;
                        host_l2[2*i] = -factor2;
                        host_l2[2*i+1] = FT(0); 
                }
                FT factor1 = -w1;
                host_u0[nLU-1] = FT(1) + factor1* host_u1[nLU-2];
                host_l1[nLU-2] = -factor1;

                cudaMalloc((void **) &l1, (nLU-1)*sizeof(FT));
                cudaMalloc((void **) &l2, (nLU-2)*sizeof(FT));
                cudaMemcpy(l1, host_l1, (nLU-1)*sizeof(FT), cudaMemcpyHostToDevice);
                cudaMemcpy(l2, host_l2, (nLU-2)*sizeof(FT), cudaMemcpyHostToDevice);

                cudaMalloc((void **) &u0,  nLU*sizeof(FT));
                cudaMalloc((void **) &u1, (nLU-1)*sizeof(FT));
                cudaMalloc((void **) &u2, (nLU-2)*sizeof(FT));

                cudaMemcpy(u0, host_u0, nLU*sizeof(FT), cudaMemcpyHostToDevice);
                cudaMemcpy(u1, host_u1, (nLU-1)*sizeof(FT), cudaMemcpyHostToDevice);
                cudaMemcpy(u2, host_u2, (nLU-2)*sizeof(FT), cudaMemcpyHostToDevice);

                delete[] host_l1;
                delete[] host_l2;
                delete[] host_u0;
                delete[] host_u1;
                delete[] host_u2;

                cudaMalloc((void **) &b2, m*nLU *sizeof(FT));
                cudaMalloc((void **) &Ux2, m*nLU * sizeof(FT));
                cudaMalloc((void **) &x2, m*nLU * sizeof(FT));

                cudaMalloc((void **) &c, m*m*sizeof(FT));
                cudaMalloc((void **) &c_trans, m*m*sizeof(FT));
         
                cudaMalloc((void **) &y, m*m*sizeof(FT));

                cudaMalloc((void **) &b_trans, m*m*sizeof(FT));
                
                int minGridSize;

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &thomas_kernel_block_size,
                                                   thomas_kernel<FT>, 0, m*num_blocks);

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &ST_block_borders_kernel_block_size,
                                                   ST_block_borders_kernel<FT>, 0, m*nLU);

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &ST_lu_forwards_kernel_block_size, 
                                                   ST_lu_forwards_kernel<FT>, 0, m); 

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &ST_lu_backwards_kernel_block_size, 
                                                   ST_lu_backwards_kernel<FT>, 0, m); 

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &ST_backsubstitution_kernel_block_size, 
                                                   ST_backsubstitution_kernel<FT>, 0, m*m); 
        }


        virtual void run(FT const * const b, FT * const x) {  
                cublas_transpose(cublas_handle, m, b, b_trans);
                run2(b_trans, c_trans);
                cublas_transpose(cublas_handle, m, c_trans, c);
                run2(c, x);
        }


private:
        void run2(FT * const b, FT * const x) {
                int block_count;
                int threads_per_block;

                block_count =  divide_and_round_up(m*num_blocks, thomas_kernel_block_size); 
                threads_per_block = thomas_kernel_block_size;
                thomas_kernel<FT><<<block_count, threads_per_block>>>(m, block_size, num_blocks, alpha, beta, c_prime, b, y);

                block_count =  divide_and_round_up(m*nLU, ST_block_borders_kernel_block_size); 
                threads_per_block = ST_block_borders_kernel_block_size;
                ST_block_borders_kernel<FT><<<block_count, threads_per_block>>>(m, block_size, nLU, y, b2);

                block_count =  divide_and_round_up(m, ST_lu_forwards_kernel_block_size); 
                threads_per_block = ST_lu_forwards_kernel_block_size;
                ST_lu_forwards_kernel<FT><<<block_count, threads_per_block>>>(m, nLU, l1, l2, b2, Ux2);

                block_count =  divide_and_round_up(m, ST_lu_backwards_kernel_block_size); 
                threads_per_block = ST_lu_backwards_kernel_block_size;
                ST_lu_backwards_kernel<FT><<<block_count, threads_per_block>>>(m, nLU, u0, u1, u2, Ux2, x2);

                block_count =  divide_and_round_up(m*m, ST_backsubstitution_kernel_block_size); 
                threads_per_block = ST_backsubstitution_kernel_block_size;
                ST_backsubstitution_kernel<FT><<<block_count, threads_per_block>>>(m, block_size, nLU, left_spike, right_spike, y, x2, x); 
        }

public:
        ~SpikeThomasPreconditioner() {
                cudaFree(c_prime);
                cudaFree(l1);
                cudaFree(l2);
                cudaFree(u0);
                cudaFree(u1);
                cudaFree(u2);
                // TODO more
        }
};


/*
 *      Preconditioner for the 2D problem. This uses the PCR algorithm from cusparse.
 *
 *      @param FT Field Type - Either float or double
 *      
 */ 
template<class FT>
class CusparsePCRPreconditioner : public Preconditioner<FT> {
private:
        int m;
        cusparseHandle_t cusparse_handle;
        cublasHandle_t cublas_handle;
        FT *dl, *d, *du;
        FT *d_x_trans;
public:
        virtual void init(int const m, FT const alpha, cublasHandle_t cublas_handle, cusparseHandle_t cusparse_handle) {
               
                this->m = m;
                this->cusparse_handle = cusparse_handle;
                this->cublas_handle = cublas_handle;

                cudaMalloc((void **) &dl, m*sizeof(FT));
                cudaMalloc((void **) &d, m*sizeof(FT));
                cudaMalloc((void **) &du, m*sizeof(FT));

                FT beta = sqrt(alpha);

                device_memset<FT>(d, (alpha + FT(2)) / beta, m, 0);
                device_memset<FT>(dl, FT(0), 1, 0);
                device_memset<FT>(dl, FT(-1)/beta, m-1, 1);
                device_memset<FT>(du, FT(0), 1, m-1);
                device_memset<FT>(du, FT(-1)/beta, m-1, 0);

                cudaMalloc((void **) &d_x_trans, m*m * sizeof(FT)); 
        }

        virtual void run(FT const * const d_b, FT * const d_x) {
                cublas_copy(cublas_handle, m*m, d_b, d_x);

                cusparse_gtsv(cusparse_handle, m, m, dl, d, du, d_x);

                cublas_transpose(cublas_handle, m, d_x, d_x_trans);

                cusparse_gtsv(cusparse_handle, m, m, dl, d, du, d_x_trans);
               
                cublas_transpose(cublas_handle, m, d_x_trans, d_x);
        }

        ~CusparsePCRPreconditioner() {
                cudaFree(dl);
                cudaFree(d);
                cudaFree(du);

                cudaFree(d_x_trans);
        }
};


