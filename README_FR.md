Sommaire :
==========

1.  Résultats attendus

2.  Démarche générale

3.  Prérequis

4.  CUDA : création du programme de calcul avec plusieurs GPU

5.  PYTHON 1 : Création des images taille réel avec plusieurs GPU

6.  PYTHON 2 : Création DZI

7.  WEB : Création site WEB

# 1. Résultats Attendus

Reproduction du site web : [https://fractals-julia.com/](https://fractals-julia.com/)

On chercher à obtenir des fractale de Julia en deux formats d’image avec une taille importante (&gt;32k px de coté) :

-   Bi couleur (N&B)

-   En nuance de gris

<img src=".//media/image1.png" style="width:2.76067in;height:2.5297in" alt="Une image contenant Graphique, art, symbole, cercle Description générée automatiquement" />

Figure : Image bi-couleurs de la fractale de Julia

<img src=".//media/image2.png" style="width:2.72986in;height:2.72986in" alt="Une image contenant ciel, obscurité, nuage, noir Description générée automatiquement" />

Figure : Image en nuance de gris de la fractale de Julia

De même on, utilisera un site web locale ou sur réseaux pour visualiser les fractales de Julia, l’interface tel que :

<img src=".//media/image3.png" style="width:6.3in;height:4.50556in" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

Figure : Interface WEB

Description de l’interface WEB :

1.  A => Titre Dynamique avec les valeurs de X et Y

2.  B => Axe des X : permet de modifier la valeur de X

3.  C => Axe des Y : permet de modifier la valeur de Y

4.  D => Option d’affichage et bouton de téléchargement de l’image d’origine

5.  E => Explorateur de la fractale, avec un zoom important possible.

# 2. Démarche générale

Il y a 4 étapes à respecte :

## 1.  Création d’un tableau du nombre d’itération de chaque pixel de l’image

    -   Outil : CUDA (C / C++)

    -   OS : Linux (WSL 2 Ubuntu)

    -   Matériel : Carte graphique Nvidia  8 Go RAM

## 2.  Transformation du tableau du nombre d’itération en images et compression du tableau pour optimise l’usage du disque dur.

    -   Outil : CUDA (C / C++) et python 3

    -   OS : Linux (WSL 2 Ubuntu)

    -   Matériel : Carte graphique Nvidia  8 Go RAM

## 3.  Création d’image zoomable avec le logiciel « openseadragon » et « deepzoom.py »

    -   Outil : python 3

    -   OS : Linux (WSL 2 Ubuntu) ou Windows

## 4.  Création du site web pour visualiser les fractales

    -   Outil : python 3 / HTML / JS

    -   OS : Linux (WSL 2 Ubuntu) ou Windows

# 3. Prérequis


## 1. Activer WSL2 et NVIDIA (documentation)

[https://learn.microsoft.com/fr-fr/windows/ai/directml/gpu-cuda-in-wsl](https://learn.microsoft.com/fr-fr/windows/ai/directml/gpu-cuda-in-wsl)

[https://docs.nvidia.com/cuda/wsl-user-guide/index.html](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)


## 2. Commandes Ubuntu
```bash
# Installation des Pilotes et Toolkits NVIDIA
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
sudo apt-get -y install cuda-tools-12-4
sudo apt-get -y install cuda-runtime-12-4
sudo apt-get -y install cuda-12-4

# Installation pip3
sudo apt install python3-pip

# Installation des Librairies Python
pip3 install numpy numba pillow joblib py7zr
```

## 3 Estimer les paramètres des calculs

On utilise la fonction 3D des calcules de la carte graphique donc on découpe en tuile l'image tel que :

- taille de la tuile est un multiple de 256
 
- Le nombre de tuiles par coté est defini tel que : floor(sqrt(taille de la tuile)) :

<img src=".//media/G1.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

- La coté de l'image fait donc : (nombre de tuiles par coté) * (taille de la tuile)

<img src=".//media/G2.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

- On ainsi la taille de l'image non compresé : 

<img src=".//media/G3.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

- et aussi la taille du binaire non compresé en int (32 bits) : 

<img src=".//media/G4.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

- et aussi la taille du binaire non compresé en long (64 bits) : 

<img src=".//media/G5.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />


Comme les cartes graphiques disponible ont maximun 80 Go de RAM :

- Pour des calculs en INT (32 bits) :

<img src=".//media/G4-zoom.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

- Pour des calculs en LONG (64 bits) :

<img src=".//media/G5-zoom.png" alt="Une image contenant texte, capture d’écran Description générée automatiquement" />

# 4. CUDA : création du programme de calcul avec plusieurs GPU


Le code cuda permet d’utiliser les GPU NVDIA comme centre de calculs.

Le code que je propose est décompose en 5 parties :

## 1.  Le header

C’est le code commun entre le code cuda et c++, on y trouve :

-   Le type de fractale à générer : Type\_Fractal

```c++
// Définition de l'énumération pour le type de fractale
enum Type_Fractal { Mandelbrot, Julia };
```

-   La structure **Complex** pour représenter les nombres complexes

```c++
// Définition de la structure Complex pour représenter les nombres complexes
struct Complex
{
    double x, y; // Partie réelle et imaginaire

    // Constructeur pour initialiser un nombre complexe
    __host__ __device__
    Complex(double a = 0.0, double b = 0.0) : x(a), y(b) {}

    // Surcharge de l'opérateur + pour l'addition de deux nombres complexes
    __host__ __device__
    Complex operator+(const Complex &other) const
    {
        return Complex(x + other.x, y + other.y);
    }

    // Surcharge de l'opérateur - pour la soustraction de deux nombres complexes
    __host__ __device__
    Complex operator-(const Complex &other) const
    {
        return Complex(x - other.x, y - other.y);
    }

    // Surcharge de l'opérateur * pour la multiplication de deux nombres complexes
    __host__ __device__
    Complex operator*(const Complex &other) const
    {
        return Complex(x * other.x - y * other.y, x * other.y + y * other.x);
    }

    // Fonction pour calculer la norme d'un nombre complexe
    __host__ __device__ double norm() const
    {
        return sqrt(x * x + y * y);
    }

    // Fonction pour élever un nombre complexe à une puissance donnée
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

-   La structure **ParameterPicture** pour stocker les paramètres de l'image fractale

```c++
// Définition de la structure ParameterPicture pour stocker les paramètres de l'image fractale
struct ParameterPicture
{
    long lenG; // Longueur globale en 3D
    long lenL; // Longueur locale en 2D
    double2 start; // Point de départ de l'image
    double size; // Taille d'un côté de l'image
    Type_Fractal type_fractal; // Type de fractale (Mandelbrot ou Julia)
    double2 coef_julia; // Coefficients pour la fractale de Julia
    double power_value; // Valeur de la puissance
    long iter_max; // Nombre maximal d'itérations
    long id; // Identifiant de l'image

    // Constructeur pour initialiser un objet ParameterPicture
    __host__ __device__ ParameterPicture(long id, long lenG, double2 start, double size, double power_value, long iter_max, Type_Fractal type_fractal, double2 coef_julia = make_double2(0.0, 0.0)) 
        : id(id), power_value(power_value), iter_max(iter_max), type_fractal(type_fractal), coef_julia(coef_julia), lenG(lenG), lenL(floorf(sqrtf((float)lenG))), start(start), size(size) {};

    // Fonction pour obtenir la taille de l'image en 3D
    __host__ __device__ size_t Get_size_array_3D() const
    {
        return (size_t)lenG * (size_t)lenG * (size_t)lenG;
    }

    // Fonction pour obtenir la taille de l'image en 2D
    __host__ __device__ size_t Get_size_array_2D() const
    {
        return (size_t)lenG * (size_t)lenG * (size_t)lenL * (size_t)lenL;
    }

    // Fonction pour obtenir la position en coordonnées double dans l'image
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

    // Fonction pour obtenir la position en coordonnées long dans l'image
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

    // Fonction pour obtenir l'index 3D d'une position dans l'image
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

    // Fonction pour obtenir l'index 2D d'une position dans l'image
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

    // Fonction pour définir une valeur dans les données de l'image à une position donnée
    __host__ __device__ void Set_Value(int x, int y, int z, long *data, long value) const
    {
        long index = Get_index_2D(x, y, z);
        if (index >= 0)
        {
            data[index] = value;
        }
    }

    // Fonction pour obtenir une valeur des données de l'image à une position donnée
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

    // Fonction pour imprimer les paramètres de l'image dans un fichier
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

## 2.  Le code cuda

C’est le code qui calcul la fractale de Julia ou de Mandelbrot, on y
trouve :

-   Kernel\_Picture : Kernel CUDA pour générer une image fractale

```c++
// Kernel CUDA pour générer une image fractale
__global__ void Kernel_Picture(ParameterPicture parameter_picture, long *data)
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
        
        long iter = 0;

        // Calculer le nombre d'itérations pour la fractale
        while (z.norm() < 2.0 && iter < parameter_picture.iter_max)
        {
            z = z.power(parameter_picture.power_value) + c;
            iter++;
        }

        // Stocker le nombre d'itérations dans le tableau de données
        data[index] = iter;
    }
}
```

-   RUN : la fonction pour exécuter le kernel CUDA

```c++
// Fonction pour exécuter le kernel CUDA
cudaError_t RUN(ParameterPicture parameter_picture, long *datas, int id_cuda)
{
    // Calculer la taille des données à allouer
    size_t size = parameter_picture.Get_size_array_2D() * sizeof(long);
    long *dev_datas = 0;
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
    Kernel_Picture<<<numBlocks, threadsPerBlock>>>(parameter_picture, dev_datas);

    // Vérifier si le lancement du kernel a échoué
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel_Picture launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Attendre la fin de l'exécution du kernel
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel_Picture!\n", cudaStatus);
        goto Error;
    }

    // Copier les données du GPU vers la mémoire de l'hôte
    cudaStatus = cudaMemcpy(datas, dev_datas, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Libérer la mémoire allouée sur le GPU
    cudaFree(dev_datas);

    // Réinitialiser le GPU
    cudaStatus = cudaDeviceReset();
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
```

## 3.  Le code C++

C’est le code qui permet de gérer la création de fractales de Julia ou
de Mandelbrot, on y trouve :

-   File\_Generate : la structure pour gérer les fichiers (.bin et .txt)

```c++
// Structure pour générer des fichiers
struct File_Generate
{
    std::string bin, txt; // Chemins des fichiers binaires et texte
    bool exist;           // Indicateur si le fichier existe
    File_Generate(std::string bin, std::string txt) : bin(bin), txt(txt) {}
};
```

-   RUN : Déclaration de la fonction CUDA externe

```c++
// Déclaration de la fonction CUDA externe
extern cudaError_t RUN(ParameterPicture parameter_picture, long *datas, int id_cuda);
```

-   CreateFolder : Fonction pour créer le dossier de travail

```c++
// Fonction pour créer le dossier de travail
std::string CreateFolder(std::string name, std::string dirBase)
{
    std::string dirNameBase = dirBase;
    std::string dirName = dirNameBase + "/" + name;

    mkdir(dirNameBase.c_str(), 0777);
    if (mkdir(dirName.c_str(), 0777) == 0)
    { // Note : 0777 donne les droits d'accès rwx pour tous
        std::cout << "Directory created: " << dirName << std::endl;
    }
    else
    {
        std::cout << "Failed to create directory!" << std::endl;
    }

    return dirName;
}
```

-   if\_file\_exist : Fonction pour vérifier si un fichier existe

```c++
// Fonction pour vérifier si un fichier existe
bool if_file_exist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}
```

-   write\_bin : Fonction pour écrire des données binaires dans un
    fichier

```c++
// Fonction pour écrire des données binaires dans un fichier
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

-   run : Fonction supervision pour lancement de calculs d'une fractale

```c++
// Fonction supervision pour lancement de calculs d'une fractale 
File_Generate run(ParameterPicture parameter_picture, std::string baseDir, int id_cuda)
{
    // Création des chemins des fichiers
    std::string path_dir = CreateFolder("id_" + std::to_string(parameter_picture.id), baseDir);
    std::string path_txt = path_dir + "/parameters.txt";
    std::string path_bin = path_dir + "/data.bin";

    // Initialisation de la structure File_Generate
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

-   Get\_nbfiles\_bin : Fonction pour obtenir le nombre de fichiers
    binaires existants

```c++
// Fonction pour obtenir le nombre de fichiers binaires existants
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

-   Open\_file\_txt : Fonction pour ouvrir un fichier texte et lire son
    contenu

```c++
/ Fonction pour ouvrir un fichier texte et lire son contenu
std::string Open_file_txt(std::string path_file)
{
    std::string myText;
    std::string out;
    std::ifstream MyReadFile(path_file);

    while (getline(MyReadFile, myText))
    {
        out = myText;
        std::cout << path_file << " contient " << myText << std::endl;
    }

    MyReadFile.close();
    return out;
}
```

-   Main : Fonction principale qui est exécuté au lancement

```c++
int main()
{
    //coté en pixel d'une tuile, il y a int(sqrt(lenG)) de tuile par coté
    //exemple pour 720 ==> il y a  int(sqrt(720)) = 26 tuiles donc 26*720 = 18 720 px de coté soit une image de 350 438 400 px en tout
    //donc un fichier binaire de 2 803 507 200 octes soit 2.8 Go.
    const long lenG = 720;

    // nombre de fichier binaire max non traité par le scripte python
    const int max_bin_files = 4;

    //Borne min max de X
    const double coef_x_min = -1.5;
    const double coef_x_max = 1.5;

    //pas d'itération de X et Y
    const double coef_pas = 0.1;

    // Vérification de l'existence du fichier id_cuda.txt
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
        std::cout << "file not existe  " << path_file_id_cuda << std::endl;
        return 1;
    }

    // Vérification de l'existence du fichier min.txt
    std::string path_file_min = "./parameters/min.txt";
    double min_value = 0.0;
    if (if_file_exist(path_file_min))
    {
        std::string min_str = Open_file_txt(path_file_min);
        min_value = std::stod(min_str);
    }
    else
    {
        std::cout << "file not existe  " << path_file_min << std::endl;
        return 1;
    }

    // Vérification de l'existence du fichier max.txt
    std::string path_file_max = "./parameters/max.txt";
    double max_value = 0.0;
    if (if_file_exist(path_file_max))
    {
        std::string max_str = Open_file_txt(path_file_max);
        max_value = std::stod(max_str);
    }
    else
    {
        std::cout << "file not existe  " << path_file_max << std::endl;
        return 1;
    }

    std::vector<File_Generate> Files_G;

    // Construction du nom de base du répertoire
    std::string baseDir = "datas_" + id_cuda_str + "_" + std::to_string(lenG) + "p";
    long id = 0;

    // Boucles pour générer des fichiers pour différentes valeurs de coef_x et coef_y
    for (double coef_x = coef_x_min ; coef_x <= coef_x_max; coef_x += coef_pas)
    {
        for (double coef_y = min_value; coef_y < max_value; coef_y += coef_pas)
        {
            std::cout << "id =  " << id << std::endl;
            std::cout << "Get_nbfiles_bin " << Get_nbfiles_bin(Files_G) << std::endl;

            // Attente si le nombre de fichiers binaires existants dépasse la limite
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

## 4.  Le scripte pour compiler le programme.

C’est le scripte qui permet de générer l’application

```bash
/usr/local/cuda/bin/nvcc -c src/main.cu -o bin/main.o -I/usr/local/cuda/lib64 -I/usr/local/cuda/extras/CUPTI/lib64
g++ -c -I/usr/local/cuda/include  src/main.cpp -o bin/main_cpp.o 
g++ bin/main.o bin/main_cpp.o -o main -lcudart -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64
```

## 5.  Les paramètres

C’est les paramètres de calculs externes au programme, on y trouve :

-   L’id de la carte nvidia à utiliser de 0 à N, n étant le nombre -1 de
    cartes graphiques disponibles

-   La borne minimale du coef y de Julia

-   La borne maximale du coef y de Julia


## 6.  Exécution du programme

Compilation de code depuis le répertoire **01-02_Creation_Datas_et_Images_full** :

```bash
$ bash ./make_main.sh
```

Exécution du code depuis le répertoire **01-02_Creation_Datas_et_Images_full**  :

```bash
$ ./main
```

# 5. PYTHON 1 : Création des images taille réel avec plusieurs GPU

## 1. Les libraires :

```python 
# Librairie de gestion de tableau très rapide
import numpy 
# Librairie de gestion de création de fonction cuda directement en python
from numba import jit, cuda
# Librairie de création et de manipulation d'images
from PIL import Image
# Librairie de gestion de chronomètre
import time
# Librairie de création de threads
from joblib import Parallel, delayed
# Librairies de base
import os
from math import *
import gc
# Librairies de création de fichiers 7zip
import py7zr
```

## 2. Les constantes :

```python 
# Nombre de Thread lancé
NB_THREAD = 1
#Noms des fichiers constant
FILE_TXT = "parameters.txt"
FILE_BIN = "data.bin"
FILE_ZIP = "data.zip"
FILE_7ZIP = "data.7z"
FILE_NB_TIF= "out1.tif"
FILE_G_TIF= "out2.tif"
```

## 3. La class ParameterPicture et son lecteur  :

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
        exec("param."+line)
    f.close()
    return param
```

## 4. les fonctions cuda  :

```python
# Transforme un tableau d'itérations en image bi-colore (N & B) 
@jit(nopython=True)
def lineariser_2_cuda(input_array):
    # Initialisation du tableau de sortie avec le meme shape que l'entree
    output_array = numpy.empty_like(input_array, dtype=numpy.uint8)
    
    # Parcours du tableau d'entree
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            # Application de la transformation out = in % 2 * 255
            output_array[i, j] = (input_array[i, j] % 2) * 255
    
    return output_array

# Transforme un tableau d'itérations en image en nueance de gris [0 -> 255]
@jit(nopython=True)
def lineariser_255_cuda(input_array, iter_max:int):
    # Initialisation du tableau de sortie avec le meme shape que l'entree
    output_array = numpy.empty_like(input_array, dtype=numpy.uint8)

    coef = ceil(iter_max/255) 

    # Parcours du tableau d'entree
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            # Récupération de la valeur de case
            value = (input_array[i, j])
            # Assurez-vous que la valeur est dans la plage
            value = max(0, min(iter_max, value))
            # Normalisez la valeur dans l'intervalle [0, 1]
            normalized = (value) / (iter_max)
            # Ajout de coef
            phase = normalized *  coef
            # Application de la transformation 
            output_array[i, j] = int(phase*255) %255
    
    return output_array
```
## 5. les fonctions python  :

- save_tif : Création d'une image tif sur le disque dur.

```python
def save_tif(image, filename):
    # Sauvegarder l'image en TIFF avec compression
    image.save(filename, format='TIFF', compression='tiff_lzw')
```

- lister_paths_bin : liste les fichiers binaires d'un dossier.

```python
def lister_paths_bin(baseDir:str,nb_thread:int,no_thread:int):
    liste_paths= list()
    for path, subdirs, files in os.walk(baseDir):
        for name in files:
            if name==FILE_BIN:
                num = int(path.split("id_")[-1])
                if num % nb_thread == no_thread:
                    liste_paths.append(path)
    return liste_paths
```

- bin_file2image_np_2D : tranforme un fichier binaire en tableau 2D.

```python
def bin_file2image_np_2D(file_bin:str,param:ParameterPicture):
    data = numpy.zeros(0)
    start_l = time.time()
    with open(file_bin, 'rb') as f:
        data = numpy.fromfile(f, dtype=numpy.uint64)
    elapsed = time.time() - start_l
    print(f'Temps d\'execution  Open : {elapsed:.4} s')

    start_l = time.time()
    data = data.reshape(param.lenG*param.lenL,param.lenG*param.lenL)
    elapsed = time.time() - start_l
    print(f'Temps d\'execution  1d --> 2D : {elapsed:.4} s')
    return data
```

## 6. la fonctions principale  :

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
                    print(f"Use : {file_txt}")
                    param = generate_ParameterPicture(file_txt)
                    end = time.time()
                    elapsed = end - start
                    print(f'Temps d\'execution  generate_ParameterPicture: {elapsed:.4} s')
                    
                    ########################### bin_file2image_np_2D ######################
                    start = time.time()
                    image_raw = bin_file2image_np_2D(file_bin,param)
                    end = time.time()
                    elapsed = end - start
                    print(f'Temps d\'execution  bin_file2image_np: {elapsed:.4} s')
                    max = numpy.max(numpy.max(image_raw))
                    gc.collect()

                    ########################### create img N&B  % 2  ######################
                    print("create img N&B  % 2 ")

                    start = time.time()
                    image_np_lineariser = lineariser_2_cuda(image_raw)
                    end = time.time()
                    elapsed = end - start
                    print(f'Temps d\'execution  lineariser_2_cuda: {elapsed:.4} s')

                    im_out = Image.fromarray(image_np_lineariser.astype('uint8')).convert('L')
                    save_tif(im_out,file_nb_tif)
                        
                    del image_np_lineariser
                    del im_out
                    gc.collect()

                    ########################### create img N&B  % 255  ######################
                    print("create img N&B  % 255 ")

                    start = time.time()
                    image_np_lineariser = lineariser_255_cuda(image_raw,max)
                    end = time.time()
                    elapsed = end - start
                    print(f'Temps d\'execution  lineariser_255_cuda: {elapsed:.4} s')
                    

                    im_out = Image.fromarray(image_np_lineariser.astype('uint8')).convert('L')
                    save_tif(im_out,file_g_tif)

                    del image_np_lineariser
                    del im_out
                    del image_raw
                    gc.collect()

                    ########################### create 7zip datas  ######################

                    print("Create a 7ZipFile Object")
                    start =  time.time()
                    with py7zr.SevenZipFile(file_7zip, 'w') as z:
                        z.writeall(file_bin)
                    elapsed = time.time() - start
                    print(f'Temps d\'execution  7zip : {elapsed:.5} s')
 
                    # Check to see if the zip file is created
                    if os.path.exists(file_7zip):
                        print("7ZIP file created")
                        os.remove(file_bin)
                    else:
                        print("7ZIP file not created")
                    elapsed_l = time.time() - start_l
                    print("***************************************************************************")
                    print(f'Temps d\'execution  {path_bin}: {elapsed_l:.5} s')
                    print("***************************************************************************")
            
```

## 6. la dispatcher de la fonction principale  :

```python
def main_bin_2_tif():
    f = open("parameters/id_cuda.txt","r")
    id_cuda = int(f.readline())
    f.close()
    print(f" id cuda = {id_cuda}")
    cuda.select_device(id_cuda)

    nb_thread = NB_THREAD
    values = range(0,nb_thread)
    Parallel(n_jobs=nb_thread,prefer="threads")(delayed(sub_main_bin_2_tif)(value) for value in values)
```

## 7. lancement du scripte  :

```python
if __name__ == "__main__":
    start_g = time.time()
    main_bin_2_tif()
    end_g = time.time()
    elapsed = end_g - start_g
    print(f'Temps d\'execution  G: {elapsed} s')
```


# 6.  PYTHON 2 : Création DZI

On se base sur [deepzoom3](https://github.com/muranamihdk/deepzoom3) mise à jour vis à vis des évolution de la librairie PIllow.

[deepzoom3](https://github.com/muranamihdk/deepzoom3) permet de transformer une image de grande taille en tuiles de petites tailles (1024 px) qui permettent d'optimiser l'usage sur un site web. un peu comme google map.

le moteur web qui sera utilisé est [openseadragon](https://openseadragon.github.io/)

le code qui réalise cette tache est **Sub_03_Export_Web.py** avec la librairie **lib_deepzoom.py**.



# 7.  WEB : Création site WEB 

le code qui réalise cette tache est **Sub_04_Index_DZI.py**

Il crée un fichier *.js* qui contient la liste des fichiers *.dzi* et des images tailles réelles.

les limites des axes / ranges sont calculées automatiquement, il reste à mettre à la main le pas (step) dans le fichier **index.html**.

```html
<!-- axe X -->
<input type="range" class="custom-range" id="range_x"  step="50" onchange="update_plot()"
value="10">

<!-- axe Y -->
<input type="range" class="custom-range" id="range_y" step="50" onchange="update_plot()"
value="10">
```

Pour lancer le serveur web locale, il faut exécuter le scripte **run_web.sh** dans le dossier **./03-04_Export_WEB_et_site_web/Web** et ouvrir l'url [http://localhost:8000/](http://localhost:8000/)
