TEST Follow : https://cuda.keter.top/impl_matmul/
参考：1.https://zhuanlan.zhihu.com/p/657632577
matmul_raw.cu : naive gemm
matmul_sharedMem.cu : use shared memory for optimization 减少对全局内存的访问使用
matmul_1Dtiled.cu : 一个线程块中的线程按一维的方式进行划分，每个线程负责计算一部分的数据；减少线程块的数量，从而减少线程块的同步开销
matmul_2Dtiled.cu : 把矩阵C等分为BM*BN大小的分块，每个分块由一个block计算，其中每个thread负责计算矩阵C中TM*TN个元素