# Contents :

1.  [Expected results](https://github.com/renaudfractale/Public_Fractal_Julia#1-expected-results)

2.  [General approach](https://github.com/renaudfractale/Public_Fractal_Julia#2-general-approach)

3.  [Prerequisites](https://github.com/renaudfractale/Public_Fractal_Julia#3-prerequisites)

4.  [CUDA: creating a calculation program with several GPUs](https://github.com/renaudfractale/Public_Fractal_Julia#4-cuda-creating-a-calculation-program-with-several-gpus)

5.  [PYTHON 1: Creating life-size images with multiple GPUs](https://github.com/renaudfractale/Public_Fractal_Julia#5-python-1-creating-life-size-images-with-multiple-gpus)

6.  [PYTHON 2: DZI creation](https://github.com/renaudfractale/Public_Fractal_Julia#6-python-2-dzi-creation)

7.  [WEB : WEB site creation](https://github.com/renaudfractale/Public_Fractal_Julia#7-web--web-site-creation)

# 1. Expected results

Website reproduction: https:[//fractals-julia.com/](https://fractals-julia.com/)

We are looking to obtain Julia fractals in two image formats with a large size (&gt;32k px sides):

-   Bi color (B&W)

-   In shades of grey

<img src=".//media/image1.png" style="width:2.76067in;height:2.5297in" alt="Une image contenant Graphique, art, symbole, cercle Description générée automatiquement" />
Figure: Two-colour image of the Julia fractal

<img src=".//media/image2.png" style="width:2.72986in;height:2.72986in" alt="Une image contenant ciel, obscurité, nuage, noir Description générée automatiquement" />

Figure: Grayscale image of the Julia fractal

Similarly, we'll use a local or networked website to visualize Julia's fractals, with an interface such as :

<img src=".//media/image3.png" style="width:6.3in;height:4.50556in" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />
Figure: WEB interface

WEB interface description :

1.  A =&gt; Dynamic title with X and Y values

2.  B =&gt; X axis: allows you to change the value of X

3.  C =&gt; Y axis: allows you to change the Y value

4.  D =&gt; Display option and original image download button

5.  E =&gt; Fractal explorer, with high zoom capability.

# 2. General approach

There are 4 steps to follow:

## 1. Create a table of the number of iterations for each pixel in the image

    - Tool: CUDA (C / C++)

    - OS: Linux (WSL 2 Ubuntu)

    - Hardware: Nvidia 8 GB RAM graphics card

## 2. Transformation of the iteration number array into images and compression of the array to optimize hard disk usage.


    - Tools: CUDA (C / C++) and python 3

    - OS: Linux (WSL 2 Ubuntu)

    - Hardware: Nvidia 8 GB RAM graphics card

## 3. Creating a zoomable image with "openseadragon" and "deepzoom.py" software


    - Tool: python 3

    - OS: Linux (WSL 2 Ubuntu) or Windows

## 4. Creation of a website to visualize fractals


    - Tools: python 3 / HTML / JS

    - OS: Linux (WSL 2 Ubuntu) or Windows

# 3. Prerequisites


## 1. Activate WSL2 and NVIDIA (documentation)


<https://learn.microsoft.com/fr-fr/windows/ai/directml/gpu-cuda-in-wsl>

<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>

## 2. Ubuntu commands

```bash
    # Installing NVIDIA drivers and toolkits
    sudo apt-key del 7fa2af80
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    sudo apt-get -y install cuda-tools-12-4
    sudo apt-get -y install cuda-runtime-12-4
    sudo apt-get -y install cuda-12-4

    # Pip3 installation
    sudo apt install python3-pip

    # Installing Python libraries
    pip3 install numpy numba pillow joblib py7zr
```

## 3 Estimating calculation parameters

We use the graphics card's 3D calculation function to tile the image as
follows:

-   tile size is a multiple of 256

-   The number of tiles per side is defined as: floor(sqrt(tile size)) :

<img src=".//media/G1.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

-   The dimension of the image is therefore: (number of tiles per side) \* (tile size)

<img src=".//media/G2.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

-   This is the size of the uncompressed image:

<img src=".//media/G3.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

-   and also the size of the uncompressed binary in int (32 bits) :

<img src=".//media/G4.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

-   and also the size of the long uncompressed binary (64 bits):

<img src=".//media/G5.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

As the graphics cards available have a maximum of 80 GB RAM :

-   For INT calculations (32 bits) :

<img src=".//media/G4-zoom.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

-   For LONG (64-bit) calculations :

<img src=".//media/G5-zoom.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

# 4. CUDA: creating a calculation program with several GPUs


The cuda code enables NVDIA GPUs to be used as computing centers.

The code I propose is broken down into 5 parts:

## 1. The header

This is the common code between cuda and c++, and includes :

-   The type of fractal to generate: Type\_Fractal

```C++
    // Definition of enumeration for fractal type
    enum Type_Fractal { Mandelbrot, Julia };
```

-   The **Complex** structure for representing complex numbers

```C++
    // Definition of the Complex structure to represent complex numbers
    struct Complex
    {
        double x, y; // Real and imaginary parts

        // Constructor for initializing a complex number
        __host__ __device__
        Complex(double a = 0.0, double b = 0.0) : x(a), y(b) {}

        // Overload operator + for addition of two complex numbers
        __host__ __device__
        Complex operator+(const Complex &other) const
        {
            return Complex(x + other.x, y + other.y);
        }

        // Overload operator - for subtraction of two complex numbers
        __host__ __device__
        Complex operator-(const Complex &other) const
        {
            return Complex(x - other.x, y - other.y);
        }

        // Overload the * operator to multiply two complex numbers
        __host__ __device__
        Complex operator*(const Complex &other) const
        {
            return Complex(x * other.x - y * other.y, x * other.y + y * other.x);
        }

        // Function for calculating the norm of a complex number
        __host__ __device__ double norm() const
        {
            return sqrt(x * x + y * y);
        }

        // Function to raise a complex number to a given power
        __host__ __device__
        Complex power(double p) const
        {
            double radius = sqrt(x * x + y * y);
            double angle = atan2(y, x);
            double radius_p = pow(radius, p);
            double angle_p = p * angle;

            return Complex(radius_p * cos(angle_p), radius_p * sin(angle_p));
        }
    };
```

-   **ParameterPicture** structure for storing fractal image parameters

```C++

    // Define ParameterPicture structure to store fractal image parameters
    struct ParameterPicture
    {
        long lenG; // Global length in 3D
        long lenL; // Local length in 2D
        double2 start; // Image starting point
        double size; // Size of one side of the image
        Type_Fractal type_fractal; // Fractal type (Mandelbrot or Julia)
        double2 coef_julia; // Julia fractal coefficients
        double power_value; // Power value
        long iter_max; // Maximum number of iterations
        long id; // Image identifier

        // Constructor to initialize a ParameterPicture object
        __host__ __device__ ParameterPicture(long id, long lenG, double2 start, double size, double power_value, long iter_max, Type_Fractal type_fractal, double2 coef_julia = make_double2(0.0, 0.0)) 
            : id(id), power_value(power_value), iter_max(iter_max), type_fractal(type_fractal), coef_julia(coef_julia), lenG(lenG), lenL(floorf(sqrtf((float)lenG))), start(start), size(size) {};

        // Function to obtain 3D image size
        __host__ __device__ size_t Get_size_array_3D() const
        {
            return (size_t)lenG * (size_t)lenG * (size_t)lenG;
        }

        // Function to obtain 2D image size
        __host__ __device__ size_t Get_size_array_2D() const
        {
            return (size_t)lenG * (size_t)lenG * (size_t)lenL * (size_t)lenL;
        }

        // Function to obtain the position in double coordinates in the image
        __host__ __device__ double2 GetPose_double(int x, int y, int z) const
        {
            int id = 0;
            for (long x_ = 0; x_ < lenL; x_++)
            {
                for (long y_ = 0; y_ < lenL; y_++)
                {
                    if (id == z)
                    {
                        return make_double2(start.x + ((double)x_ * size) + ((double)x / (double)lenG * size), start.y + ((double)y_ * size) + ((double)y / (double)lenG * size));
                    }
                    id++;
                }
            }
            return make_double2(0.0, 0.0);
        }

        // Function to obtain the position in long coordinates in the image
        __host__ __device__ long2 GetPose_long(int x, int y, int z) const
        {
            int id = 0;
            for (long x_ = 0; x_ < lenL; x_++)
            {
                for (long y_ = 0; y_ < lenL; y_++)
                {
                    if (id == z)
                    {
                        return make_long2((x_ * lenG) + (long)x, (y_ * lenG) + (long)y);
                    }
                    id++;
                }
            }
            return make_long2(0, 0);
        }

        // Function to obtain the 3D index of a position in the image
        __host__ __device__ long Get_index_3D(int x, int y, int z) const
        {
            if (x < 0 || (long)x >= lenG)
                return -1;
            if (y < 0 || (long)y >= lenG)
                return -1;
            if (z < 0 || (long)z >= lenL * lenL)
                return -1;

            return (long)z * lenG * lenG + (long)y * lenG + (long)x;
        }

        // Function to obtain the 2D index of a position in the image
        __host__ __device__ long Get_index_2D(int x, int y, int z) const
        {
            if (x < 0 || (long)x >= lenG)
                return -1;
            if (y < 0 || (long)y >= lenG)
                return -1;
            if (z < 0 || (long)z >= (lenL * lenL))
                return -1;

            long2 pose = GetPose_long(x, y, z);
            return pose.y * lenG * lenL + pose.x;
        }

        // Function to set a value in the image data at a given position
        __host__ __device__ void Set_Value(int x, int y, int z, long *data, long value) const
        {
            long index = Get_index_2D(x, y, z);
            if (index >= 0)
            {
                data[index] = value;
            }
        }

        // Function to obtain an image data value at a given position
        __host__ __device__ long Get_Value(int x, int y, int z, long *data) const
        {
            long index = Get_index_2D(x, y, z);
            if (index >= 0)
            {
                return data[index];
            }
            else
            {
                return 0;
            }
        }

        // Function to print image parameters to a file
        __host__ void print_file(std::string path_file) const
        {
            std::ofstream myfile;
            myfile.open(path_file, std::ios::app);
            myfile << "id = " << id << std::endl;

            myfile << "lenG = " << lenG << std::endl;
            myfile << "lenL = " << lenL << std::endl;

            myfile << "start_x = " << start.x << std::endl;
            myfile << "start_y = " << start.y << std::endl;

            myfile << "size = " << size << std::endl;
            myfile << "type_fractal = " << type_fractal << std::endl;
            myfile << "coef_julia_x = " << coef_julia.x << std::endl;
            myfile << "coef_julia_y = " << coef_julia.y << std::endl;

            myfile << "power_value = " << power_value << std::endl;
            myfile << "iter_max = " << iter_max << std::endl;
            myfile.close();
        }
    };
```

## 2. The cuda code


This is the code that calculates the Julia or Mandelbrot fractal:

-   Kernel\_Picture: CUDA kernel for generating a fractal image

```C++
    // CUDA kernel to generate a fractal image
    __global__ void Kernel_Picture(ParameterPicture parameter_picture, long *data)
    {
        // 3D index calculation for each thread
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int idz = blockIdx.z * blockDim.z + threadIdx.z;

        // Get the corresponding 2D index
        long index = parameter_picture.Get_index_2D(idx, idy, idz);

        // If index is valid
        if (index >= 0)
        {
            // Get the corresponding complex position
            double2 pos_double = parameter_picture.GetPose_double(idx, idy, idz);
            Complex z(pos_double.x, pos_double.y);
            Complex c(pos_double.x, pos_double.y);

            // If fractal type is Julia, use Julia coefficients
            if (parameter_picture.type_fractal == Type_Fractal::Julia)
            {
                c.x = parameter_picture.coef_julia.x;
                c.y = parameter_picture.coef_julia.y;
            }
            
            long iter = 0;

            // Calculate the number of iterations for the fractal
            while (z.norm() < 2.0 && iter < parameter_picture.iter_max)
            {
                z = z.power(parameter_picture.power_value) + c;
                iter++;
            }

            // Store the number of iterations in the data array
            data[index] = iter;
        }
    }
```

-   RUN: the function to run the CUDA kernel

```C++
    // Function to run the CUDA kernel
    cudaError_t RUN(ParameterPicture parameter_picture, long *datas, int id_cuda)
    {
        // Calculate data size to be allocated
        size_t size = parameter_picture.Get_size_array_2D() * sizeof(long);
        long *dev_datas = 0;
        cudaError_t cudaStatus;

        // Define thread and block configuration
        const dim3 threadsPerBlock(16, 16, 4);
        const dim3 numBlocks((parameter_picture.lenG + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                             (parameter_picture.lenG + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                             (parameter_picture.lenG + threadsPerBlock.z - 1) / threadsPerBlock.z);

        // Select GPU to use
        cudaStatus = cudaSetDevice(id_cuda);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        // Allocate memory on GPU for data
        cudaStatus = cudaMalloc((void **)&dev_datas, size);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        // Launch CUDA kernel
        Kernel_Picture<<numBlocks, threadsPerBlock>>(parameter_picture, dev_datas);

        // Check if kernel launch has failed
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Kernel_Picture launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Wait for kernel execution to complete
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel_Picture!\n", cudaStatus);
            goto Error;
        }

        // Copy data from GPU to host memory
        cudaStatus = cudaMemcpy(datas, dev_datas, size, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // Free memory allocated on the GPU
        cudaFree(dev_datas);

        // Reset the GPU
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceReset failed!");
            return cudaStatus;
        }

        return cudaSuccess;

    Error:
        // In the event of an error, free the memory allocated on the GPU
        cudaFree(dev_datas);
        return cudaStatus;
    }
```

## 3. C++ code


This is the code used to manage the creation of Julia or Mandelbrot
fractals:

-   File\_Generate: the structure for managing files (.bin and .txt)

```C++
    // Structure for generating files
    struct File_Generate
    {
        std::string bin, txt; // Binary and text file paths
        bool exist; // Indicator whether file exists
        File_Generate(std::string bin, std::string txt) : bin(bin), txt(txt) {}
    };
```

-   RUN: External CUDA function declaration

```C++
    // External CUDA function declaration
    extern cudaError_t RUN(ParameterPicture parameter_picture, long *datas, int id_cuda);
```

-   CreateFolder: Function to create the working folder

```C++
    // Function to create the working folder
    std::string CreateFolder(std::string name, std::string dirBase)
    {
        std::string dirNameBase = dirBase;
        std::string dirName = dirNameBase + "/" + name;

        mkdir(dirNameBase.c_str(), 0777);
        if (mkdir(dirName.c_str(), 0777) == 0)
        { // Note: 0777 gives rwx access rights for all
            std::cout << "Directory created: " << dirName << std::endl;
        }
        else
        {
            std::cout << "Failed to create directory!" << std::endl;
        }

        return dirName;
    }
```

-   if\_file\_exist: Function for checking whether a file exists

```C++
    // Function to check if a file exists
    bool if_file_exist(const std::string &name)
    {
        std::ifstream f(name.c_str());
        return f.good();
    }
```

-   write\_bin: Function for writing binary data to a file

```C++
    // Function for writing binary data to a file
    bool write_bin(std::string path_file, long *data, size_t size)
    {
        std::ofstream outfile(path_file, std::ios::out | std::ios::binary);
        if (!outfile)
        {
            std::cerr << "Cannot open file for writing.\n";
            return false;
        }

        outfile.write(reinterpret_cast<char *>(data), size * sizeof(long));
        outfile.close();

        free(data);
        return true;
    }
```

-   run: Supervision function for launching fractal calculations

```C++
    // Supervision function for launching fractal calculations 
    File_Generate run(ParameterPicture parameter_picture, std::string baseDir, int id_cuda)
    {
        // Create file paths
        std::string path_dir = CreateFolder("id_" + std::to_string(parameter_picture.id), baseDir);
        std::string path_txt = path_dir + "/parameters.txt";
        std::string path_bin = path_dir + "/data.bin";

        // Initialize File_Generate structure
        File_Generate file_generate(path_bin, path_txt);
        file_generate.exist = if_file_exist(path_txt);

        if (file_generate.exist)
            return file_generate;

        long *datas = 0;
        try
        {
            size_t size = parameter_picture.Get_size_array_2D() * sizeof(long);
            datas = (long *)malloc(size);
            cudaError_t cudaStatus;

            cudaStatus = RUN(parameter_picture, datas, id_cuda);
            if (cudaStatus == cudaSuccess)
            {
                write_bin(path_bin, datas, parameter_picture.Get_size_array_2D());
                parameter_picture.print_file(path_txt);
                file_generate.exist = true;
            }
            else
            {
                file_generate.exist = false;
            }
        }
        catch (const std::exception &)
        {
            free(datas);
            file_generate.exist = false;
            if (if_file_exist(path_txt))
                std::remove(path_txt.c_str());
            if (if_file_exist(path_bin))
                std::remove(path_bin.c_str());
        }

        return file_generate;
    }
```

-   Get\_nbfiles\_bin: Function for obtaining the number of existing
    binary files

```C++
    // Function to obtain the number of existing binary files
    int Get_nbfiles_bin(std::vector<File_Generate> Files_G)
    {
        int count = 0;
        for (File_Generate &file : Files_G)
        {
            if (file.exist)
            {
                file.exist = if_file_exist(file.bin);
                if (file.exist)
                    count++;
            }
        }
        return count;
    }
```

-   Open\_file\_txt: Function to open a text file and read its contents

```C++
    / Function to open a text file and read its contents
    std::string Open_file_txt(std::string path_file)
    {
        std::string myText;
        std::string out;
        std::ifstream MyReadFile(path_file);

        while (getline(MyReadFile, myText))
        {
            out = myText;
            std::cout << path_file << "contains" << myText << std::endl;
        }

        MyReadFile.close();
        return out;
    }
```

-   Main: Main function executed at launch

```C++
    int main()
    {
        //int(sqrt(lenG)) is the number of tiles per side.
        //example for 720 ==> there are int(sqrt(720)) = 26 tiles so 26*720 = 18,720 px per side, i.e. a total image size of 350,438,400 px
        //therefore a binary file of 2,803,507,200 octes or 2.8 GB.
        const long lenG = 720;

        // max number of binary files not processed by the python script
        const int max_bin_files = 4;

        //Borne min max of X
        const double coef_x_min = -1.5;
        const double coef_x_max = 1.5;

        //No iteration of X and Y
        const double coef_pas = 0.1;

        // Check existence of id_cuda.txt file
        std::string path_file_id_cuda = "./parameters/id_cuda.txt";
        int id_cuda = 0;
        std::string id_cuda_str = "";
        if (if_file_exist(path_file_id_cuda))
        {
            id_cuda_str = Open_file_txt(path_file_id_cuda);
            id_cuda = std::stoi(id_cuda_str);
        }
        else
        {
            std::cout << "file not exist " << path_file_id_cuda << std::endl;
            return 1;
        }

        // Check existence of min.txt file
        std::string path_file_min = "./parameters/min.txt";
        double min_value = 0.0;
        if (if_file_exist(path_file_min))
        {
            std::string min_str = Open_file_txt(path_file_min);
            min_value = std::stod(min_str);
        }
        else
        {
            std::cout << "file not existe " << path_file_min << std::endl;
            return 1;
        }

        // Check that max.txt file exists
        std::string path_file_max = "./parameters/max.txt";
        double max_value = 0.0;
        if (if_file_exist(path_file_max))
        {
            std::string max_str = Open_file_txt(path_file_max);
            max_value = std::stod(max_str);
        }
        else
        {
            std::cout << "file not existe " << path_file_max << std::endl;
            return 1;
        }

        std::vector<File_Generate> Files_G;

        // Construction of the base directory name
        std::string baseDir = "datas_" + id_cuda_str + "_" + std::to_string(lenG) + "p";
        long id = 0;

        // Loops to generate files for different coef_x and coef_y values
        for (double coef_x = coef_x_min ; coef_x <= coef_x_max; coef_x += coef_pas)
        {
            for (double coef_y = min_value; coef_y < max_value; coef_y += coef_step)
            {
                std::cout << "id = " << id << std::endl;
                std::cout << "Get_nbfiles_bin " << Get_nbfiles_bin(Files_G) << std::endl;

                // Waits if the number of existing binary files exceeds the limit
                while (Get_nbfiles_bin(Files_G) >= max_bin_files)
                {
                    std::cout << "Get_nbfiles_bin " << Get_nbfiles_bin(Files_G) << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(60ll * 1000ll));
                }

                id++;
                ParameterPicture parameter_picture(id, lenG, make_double2(-2.0, -2.0), (2.0 * 2.0) / (double)floorf(sqrtf((float)lenG)), 2, 2024, Type_Fractal::Julia, make_double2(coef_x, coef_y));
                Files_G.push_back(run(parameter_picture, baseDir, id_cuda));
            }
        }
    }
```

## 4. The script to compile the program.

This is the script used to generate the application

```bash
    /usr/local/cuda/bin/nvcc -c src/main.cu -o bin/main.o -I/usr/local/cuda/lib64 -I/usr/local/cuda/extras/CUPTI/lib64
    g++ -c -I/usr/local/cuda/include src/main.cpp -o bin/main_cpp.o 
    g++ bin/main.o bin/main_cpp.o -o main -lcudart -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64
```

## 5. Parameters


These are the program's external calculation parameters:

-   The id of the nvidia card to use from 0 to N, where n is the number-1 of available graphics cards

-   The minimum bound of Julia's coef y

-   The upper bound of Julia's coef y

## 6. Running the program

Compilation of code from directory **01-02\_Creation\_Datas\_et\_Images\_full** :

```bash
    $ bash ./make_main_int.sh
```
Code execution from directory **01-02\_Creation\_Datas\_et\_Images\_full** :

```bash
    $ ./main
```
# 5. PYTHON 1: Creating life-size images with multiple GPUs

## 1. Booksellers :

```python
    # Fast table management library
    import numpy 
    # Library for managing the creation of cuda functions directly in python
    from numba import jit, cuda
    # Image creation and manipulation library
    from PIL import Image
    # Stopwatch management library
    import time
    # Thread creation library
    from joblib import Parallel, delayed
    # Basic libraries
    import os
    from math import *
    import gc
    # 7zip file creation libraries
    import py7zr
```

## 2. Constants :

```python
    # Number of Threads launched
    NB_THREAD = 1
    #File names constant
    FILE_TXT = "parameters.txt
    FILE_BIN = "data.bin
    FILE_ZIP = "data.zip
    FILE_7ZIP = "data.7z"
    FILE_NB_TIF= "out1.tif"
    FILE_G_TIF= "out2.tif"
```

## 3. The ParameterPicture class and its reader :

```python
    class ParameterPicture :
        id=0
        lenG=0
        lenL=0
        start_x=0
        start_y=0
        size=0
        type_fractal=0
        coef_julia_x=0
        coef_julia_y=0
        power_value=0
        iter_max=0

    def generate_ParameterPicture(path_file_txt:str):
        f = open(path_file_txt, "r")
        param = ParameterPicture()
        while True:
            line = f.readline()
            if line == "":
                break
            exec("param. "+line)
        f.close()
        return param
```

## 4. cuda functions :

```python
    # Transforms an array of iterations into a two-color image (N & B) 
    @jit(nopython=True)
    def lineariser_2_cuda(input_array):
        # Initialize output array with same shape as input
        output_array = numpy.empty_like(input_array, dtype=numpy.uint8)
        
        # Browse input array
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                # Apply transformation out = in % 2 * 255
                output_array[i, j] = (input_array[i, j] % 2) * 255
        
        return output_array

    # Transforms an array of iterations into a grayscale image [0 -> 255].
    @jit(nopython=True)
    def lineariser_255_cuda(input_array, iter_max:int):
        # Initialize output array with same shape as input
        output_array = numpy.empty_like(input_array, dtype=numpy.uint8)

        coef = ceil(iter_max/255) 

        # Browse input array
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                # Retrieve case value
                value = (input_array[i, j])
                # Make sure value is within range
                value = max(0, min(iter_max, value))
                # Normalize the value in the range [0, 1].
                normalized = (value) / (iter_max)
                # Add coef
                phase = normalized * coef
                # Apply transformation 
                output_array[i, j] = int(phase*255) %255
        
        return output_array
```

## 5. python functions :

-   save\_tif: Create a tif image on hard disk.

```python
    def save_tif(image, filename):
        # Save image as TIFF with compression
        image.save(filename, format='TIFF', compression='tiff_lzw')
```

-   lister\_paths\_bin: lists binary files in a folder.

```python
    def lister_paths_bin(baseDir:str,nb_thread:int,no_thread:int):
        list_paths= list()
        for path, subdirs, files in os.walk(baseDir):
            for name in files:
                if name==FILE_BIN:
                    num = int(path.split("id_")[-1])
                    if num % nb_thread == no_thread:
                        liste_paths.append(path)
        return liste_paths
```

-   bin\_file2image\_np\_2D: transforms a binary file into a 2D array.

```python
    def bin_file2image_np_2D(file_bin:str,param:ParameterPicture):
        data = numpy.zeros(0)
        start_l = time.time()
        with open(file_bin, 'rb') as f:
            data = numpy.fromfile(f, dtype=numpy.uint64)
        elapsed = time.time() - start_l
        print(f'Time d\'execution Open : {elapsed:.4} s')

        start_l = time.time()
        data = data.reshape(param.lenG*param.lenL,param.lenG*param.lenL)
        elapsed = time.time() - start_l
        print(f'Execution time 1d --> 2D : {elapsed:.4} s')
        return data
```

## 6. main functions :

```python
    def sub_main_bin_2_tif(value:int):
        nb_thread = NB_THREAD
        while True:
            if True:
                baseDir = "./"# makeBaseDir(nbpts)
                liste_paths = lister_paths_bin(baseDir,nb_thread,value)
                for path_bin in liste_paths:
                    file_txt = os.path.join(path_bin, FILE_TXT)
                    file_bin = os.path.join(path_bin, FILE_BIN)
                    file_7zip = os.path.join(path_bin, FILE_7ZIP)
                    file_nb_tif = os.path.join(path_bin, FILE_NB_TIF)
                    file_g_tif = os.path.join(path_bin, FILE_G_TIF)
                    
                    if os.path.isfile(file_txt) and os.path.isfile(file_bin):
                        start_l = time.time()
                        ########################### generate_ParameterPicture ######################
                        start = time.time()
                        print(f "Use : {file_txt}")
                        param = generate_ParameterPicture(file_txt)
                        end = time.time()
                        elapsed = end - start
                        print(f'Execution time generate_ParameterPicture: {elapsed:.4} s')
                        
                        ########################### bin_file2image_np_2D ######################
                        start = time.time()
                        image_raw = bin_file2image_np_2D(file_bin,param)
                        end = time.time()
                        elapsed = end - start
                        print(f'Execution time bin_file2image_np: {elapsed:.4} s')
                        max = numpy.max(numpy.max(image_raw))
                        gc.collect()

                        ########################### create img B&W % 2 ######################
                        print("create img B&W % 2 ")

                        start = time.time()
                        image_np_lineariser = lineariser_2_cuda(image_raw)
                        end = time.time()
                        elapsed = end - start
                        print(f'Execution time lineariser_2_cuda: {elapsed:.4} s')

                        im_out = Image.fromarray(image_np_lineariser.astype('uint8')).convert('L')
                        save_tif(im_out,file_nb_tif)
                            
                        del image_np_lineariser
                        del im_out
                        gc.collect()

                        ########################### create img B&W % 255 ######################
                        print("create img B&W % 255 ")

                        start = time.time()
                        image_np_lineariser = lineariser_255_cuda(image_raw,max)
                        end = time.time()
                        elapsed = end - start
                        print(f'Execution time lineariser_255_cuda: {elapsed:.4} s')
                        

                        im_out = Image.fromarray(image_np_lineariser.astype('uint8')).convert('L')
                        save_tif(im_out,file_g_tif)

                        del image_np_lineariser
                        del im_out
                        del image_raw
                        gc.collect()

                        ########################### create 7zip datas ######################

                        print("Create a 7ZipFile Object")
                        start = time.time()
                        with py7zr.SevenZipFile(file_7zip, 'w') as z:
                            z.writeall(file_bin)
                        elapsed = time.time() - start
                        print(f'7zip execution time : {elapsed:.5} s')
     
                        # Check to see if the zip file is created
                        if os.path.exists(file_7zip):
                            print("7ZIP file created")
                            os.remove(file_bin)
                        else:
                            print("7ZIP file not created")
                        elapsed_l = time.time() - start_l
                        print("***************************************************************************")
                        print(f'Execution time {path_bin}: {elapsed_l:.5} s')
                        print("***************************************************************************")           
```

## 6. main function dispatcher :


```python
    def main_bin_2_tif():
        f = open("parameters/id_cuda.txt", "r")
        id_cuda = int(f.readline())
        f.close()
        print(f" id cuda = {id_cuda}")
        cuda.select_device(id_cuda)

        nb_thread = NB_THREAD
        values = range(0,nb_thread)
        Parallel(n_jobs=nb_thread,prefer="threads")(delayed(sub_main_bin_2_tif)(value) for value in values)
```

## 7. script launch :

```python
    if __name__ == "__main__":
        start_g = time.time()
        main_bin_2_tif()
        end_g = time.time()
        elapsed = end_g - start_g
        print(f 'Execution time G: {elapsed} s')
```

# 6. PYTHON 2: DZI creation

We use [deepzoom3](https://github.com/muranamihdk/deepzoom3), which has
been updated to take account of changes in the PIllow library.

[deepzoom3](https://github.com/muranamihdk/deepzoom3) transforms a large
image into small tiles (1024 px) for optimized use on a website. a bit
like google map.

the web engine to be used is
[openseadragon](https://openseadragon.github.io/)

the code that performs this task is **Sub\_03\_Export\_Web.py** with the
**lib\_deepzoom.py** library.

# 7. WEB : WEB site creation

the code that performs this task is **Sub\_04\_Index\_DZI.py**

It creates a *.js* file containing a list of *.dzi* files and full-size
images.

Axis/range limits are calculated automatically, and all that's left is
to manually enter the step in the **index.html** file.

```html
    <!-- X axis -->
    <input type="range" class="custom-range" id="range_x" step="50" onchange="update_plot()"
    value="10">

    <!-- Y axis -->
    <input type="range" class="custom-range" id="range_y" step="50" onchange="update_plot()"
    value="10">
```

To launch the local web server, run the script **run\_web.sh** in the
folder **./03-04\_Export\_WEB\_et\_site\_web/Web** and open the url
[http://localhost:8000/.](http://localhost:8000/)
