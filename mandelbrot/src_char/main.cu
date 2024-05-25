#include "common.h"
#include <stdio.h>  // Pour fprintf et stderr
#include <stdlib.h> // Pour les fonctions standard C comme malloc
#include <stdint.h>

// Kernel CUDA pour générer une image fractale
__global__ void Kernel_Picture(ParameterPicture parameter_picture, unsigned char *data, bool type_image)
{
    // Calcul des indices 3D pour chaque thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // Obtenir l'index 2D correspondant
    long index = parameter_picture.Get_index_2D(idx, idy, idz);

    // Si l'index est valide
    if (index >= 0)
    {
        // Obtenir la position complexe correspondante
        double2 pos_double = parameter_picture.GetPose_double(idx, idy, idz);
        Complex z(pos_double.x, pos_double.y);
        Complex c(pos_double.x, pos_double.y);

        // Si le type de fractale est Julia, utiliser les coefficients de Julia
        if (parameter_picture.type_fractal == Type_Fractal::Julia)
        {
            c.x = parameter_picture.coef_julia.x;
            c.y = parameter_picture.coef_julia.y;
        }

        int iter = 0;

        // Calculer le nombre d'itérations pour la fractale
        while (z.norm() < 2.0 && iter < parameter_picture.iter_max)
        {
            z = z.power(parameter_picture.power_value) + c;
            iter++;
        }

        // Stocker le nombre d'itérations dans le tableau de données
        if (type_image)
        {
            data[index] = iter % 2 * 255;
        }
        else
        {
            data[index] = iter % 256;
        }
    }
}

// Fonction pour exécuter le kernel CUDA
cudaError_t RUN(ParameterPicture parameter_picture, unsigned char *datas, int id_cuda, bool type_image)
{
    // Calculer la taille des données à allouer
    size_t size = parameter_picture.Get_size_array_2D() * sizeof(unsigned char);
    unsigned char *dev_datas = 0;
    cudaError_t cudaStatus;

    // Définir la configuration des threads et des blocs
    const dim3 threadsPerBlock(16, 16, 4);
    const dim3 numBlocks((parameter_picture.lenG + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (parameter_picture.lenG + threadsPerBlock.y - 1) / threadsPerBlock.y,
                         (parameter_picture.lenG + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Sélectionner le GPU à utiliser
    cudaStatus = cudaSetDevice(id_cuda);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allouer de la mémoire sur le GPU pour les données
    cudaStatus = cudaMalloc((void **)&dev_datas, size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Lancer le kernel CUDA
    std::cout << "Start Kernel_Picture" << std::endl;
    Kernel_Picture<<<numBlocks, threadsPerBlock>>>(parameter_picture, dev_datas, type_image);
    std::cout << "End Kernel_Picture" << std::endl;

    std::cout << "Start Vérifier si le lancement du kernel a échoué" << std::endl;
    // Vérifier si le lancement du kernel a échoué
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel_Picture launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    std::cout << "End  Vérifier si le lancement du kernel a échoué" << std::endl;

    std::cout << "Start Attendre la fin de l'exécution du kernel" << std::endl;
    // Attendre la fin de l'exécution du kernel
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel_Picture!\n", cudaStatus);
        goto Error;
    }
    std::cout << "End Attendre la fin de l'exécution du kernel" << std::endl;

    // Copier les données du GPU vers la mémoire de l'hôte
    std::cout << "Start Copier les données du GPU vers la mémoire de l'hôte" << std::endl;
    cudaStatus = cudaMemcpy(datas, dev_datas, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    std::cout << "End Copier les données du GPU vers la mémoire de l'hôte" << std::endl;

    // Libérer la mémoire allouée sur le GPU
    std::cout << "Start Libérer la mémoire allouée sur le GPU" << std::endl;
    cudaFree(dev_datas);
    std::cout << "End Libérer la mémoire allouée sur le GPU" << std::endl;

    // Réinitialiser le GPU
    std::cout << "Start Réinitialiser le GPU" << std::endl;
    cudaStatus = cudaDeviceReset();
    std::cout << "End Réinitialiser le GPU" << std::endl;
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return cudaStatus;
    }

    return cudaSuccess;

Error:
    // En cas d'erreur, libérer la mémoire allouée sur le GPU
    cudaFree(dev_datas);
    return cudaStatus;
}
