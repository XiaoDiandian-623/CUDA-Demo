编译时需要链接到cublas库
nvcc sgemm_v1.cu -o sgemm_v1 -lcublas
执行时需要输入M K N的大小
./sgemm_v1 256 256 256