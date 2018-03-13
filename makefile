all :
	nvcc -arch sm_35 -I/usr/local/cuda/include -lcublas -lcusparse -o bin/test test.cu 

run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ./bin/test

