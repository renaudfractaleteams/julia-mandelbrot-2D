/usr/local/cuda/bin/nvcc -c src_long/main.cu -o bin/main.o -I/usr/local/cuda/lib64 -I/usr/local/cuda/extras/CUPTI/lib64
g++ -c -I/usr/local/cuda/include  src_long/main.cpp -o bin/main_cpp.o 
g++ bin/main.o bin/main_cpp.o -o main_long -lcudart -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64