gemv:矩阵*向量
编译时需要指定nvcc算力 确保计算能力在5.3以上： nvcc -o gemv gemv.cu --gpu-architecture=compute_70
编译得到的gemv文件 执行 ./gemv 1 执行fp16  ./gemv 执行fp32

row major行主序 col major列主序
vec[1,N] * mat[N,m] 按col major 计算
可参考：https://zhuanlan.zhihu.com/p/382964285