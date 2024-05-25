/usr/local/cuda/bin/nvcc -c src_int/main.cu -o bin/main.o -I/usr/local/cuda/lib64 -I/usr/local/cuda/extras/CUPTI/lib64
g++ -c -I/usr/local/cuda/include  src_int/main.cpp -o bin/main_cpp.o 
g++ bin/main.o bin/main_cpp.o -o main_int -lcudart -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64