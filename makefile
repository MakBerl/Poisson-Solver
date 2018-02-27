all :
	nvcc -arch sm_35 -I/usr/local/cuda/include -lcublas -lcusparse -o test test.cu 

run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ./test

