#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include "cublas_v2.h"
#include "cusparse_v2.h"

/**
 *      cublas_wrapper.h
 *
 *      This file contains wrappers for various cublas and cusparse functions. They are all overloadet for
 *      single and double precission types, so that we don't have to write the code twice if it should be 
 *      able to work with both kinds of floating point precission. 
 *
 *      @author Simon Schoelly
 */



void cublas_transpose(cublasHandle_t const cublas_handle, int const m, double const * const x, double * const x_trans) {
        double D_ONE(1);
        cublasStatus_t cublas_status = cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, &D_ONE, x, m, NULL, NULL, m, x_trans, m);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDgeam: " << cublas_status << std::endl;
                std::abort();
        }
}
void cublas_transpose(cublasHandle_t const cublas_handle, int const m, float const * const x, float * const x_trans) {
        float F_ONE(1);
        cublasStatus_t cublas_status = cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, &F_ONE, x, m, NULL, NULL, m, x_trans, m);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasSgeam: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_transpose2(cublasHandle_t const cublas_handle, int const n, int const m, double const * const x, double * const x_trans) {
        double D_ONE(1);
        cublasStatus_t cublas_status = cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &D_ONE, x, n, NULL, NULL, m, x_trans, m);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDgeam: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_transpose2(cublasHandle_t const cublas_handle, int const n, int const m, float const * const x, float * const x_trans) {
        float F_ONE(1);
        cublasStatus_t cublas_status = cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &F_ONE, x, n, NULL, NULL, m, x_trans, m);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasSgeam: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_copy(cublasHandle_t const cublas_handle, int const n, double const * const x, double * const y) {
        cublasStatus_t cublas_status = cublasDcopy(cublas_handle, n, x, 1, y, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDcopy: " << cublas_status << std::endl;
                std::abort();
        }
}
void cublas_copy(cublasHandle_t const cublas_handle, int const n, float const * const x, float * const y) {
        cublasStatus_t cublas_status = cublasScopy(cublas_handle, n, x, 1, y, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasScopy: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_axpy(cublasHandle_t const cublas_handle, int const n, double const * const alpha, double const * const x, double * const y) {
        cublasStatus_t cublas_status = cublasDaxpy(cublas_handle, n, alpha, x, 1, y, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDaxpy: " << cublas_status << std::endl;
                std::abort();
        }
}
void cublas_axpy(cublasHandle_t const cublas_handle, int const n, float const * const alpha, float const * const x, float * const y) {
        cublasStatus_t cublas_status = cublasSaxpy(cublas_handle, n, alpha, x, 1, y, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasSaxpy: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_dot(cublasHandle_t const cublas_handle, int const n, double const * const x, double const * const y, double * const result) {
        cublasStatus_t cublas_status = cublasDdot(cublas_handle, n, x, 1, y, 1, result);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDdot: " << cublas_status << std::endl;
                std::abort();
        }
}
void cublas_dot(cublasHandle_t const cublas_handle, int const n, float const * const x, float const * const y, float * const result) {
        cublasStatus_t cublas_status = cublasSdot(cublas_handle, n, x, 1, y, 1, result);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasSdot: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_nrm2(cublasHandle_t const cublas_handle, int const n, double const * const x, double * result) {
        cublasStatus_t cublas_status = cublasDnrm2(cublas_handle, n, x, 1, result);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDnrm2: " << cublas_status << std::endl;
                std::abort();
        }
}
void cublas_nrm2(cublasHandle_t const cublas_handle, int const n, float const * const x, float * result) {
        cublasStatus_t cublas_status = cublasSnrm2(cublas_handle, n, x, 1, result);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasSnrm2: " << cublas_status << std::endl;
                std::abort();
        }
}

void cublas_scal(cublasHandle_t const cublas_handle, int const n, double const * const alpha, double * const x) {
        cublasStatus_t cublas_status = cublasDscal(cublas_handle, n, alpha, x, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasDscal: " << cublas_status << std::endl;
                std::abort();
        }
}
void cublas_scal(cublasHandle_t const cublas_handle, int const n, float const * const alpha, float * const x) {
        cublasStatus_t cublas_status = cublasSscal(cublas_handle, n, alpha, x, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Cublas error in function cublasSscal: " << cublas_status << std::endl;
                std::abort();
        }
}

void cusparse_gtsv(cusparseHandle_t const cusparse_handle, int const m, int const n, double const * const dl,
                 double const * const d, double const * const du, double * const x) {
        cusparseStatus_t cusparse_status = cusparseDgtsv_nopivot(cusparse_handle, m, n, dl, d, du, x, m);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
                std::cout << "Cusparse error in function cusparseDgtsv_nopivot: " << cusparse_status << std::endl;
                std::abort();
        }
}
void cusparse_gtsv(cusparseHandle_t const cusparse_handle, int const m, int const n, float const * const dl,
                 float const * const d, float const * const du, float * const x) {
        cusparseStatus_t cusparse_status = cusparseSgtsv_nopivot(cusparse_handle, m, n, dl, d, du, x, m);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
                std::cout << "Cusparse error in function cusparseDgtsv_nopivot: " << cusparse_status << std::endl;
                std::abort();
        }
}

#endif
