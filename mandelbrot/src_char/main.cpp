#include "common.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <sys/stat.h> // Pour mkdir sur Unix/Linux

// Structure pour générer des fichiers
struct File_Generate
{
    std::string bin, txt; // Chemins des fichiers binaires et texte
    bool exist;           // Indicateur si le fichier existe
    File_Generate(std::string bin, std::string txt) : bin(bin), txt(txt) {}
};

// Déclaration de la fonction CUDA externe
extern cudaError_t RUN(ParameterPicture parameter_picture, unsigned char *datas, int id_cuda, bool type_image);

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

// Fonction pour vérifier si un fichier existe
bool if_file_exist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

// Fonction pour écrire des données binaires dans un fichier
bool write_bin(std::string path_file, int *data, size_t size)
{
    std::ofstream outfile(path_file, std::ios::out | std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Cannot open file for writing.\n";
        return false;
    }

    outfile.write(reinterpret_cast<char *>(data), size * sizeof(int));
    outfile.close();

    free(data);
    return true;
}

// Fonction pour écrire des données binaires dans un fichier
bool write_bin_char(std::string path_file, unsigned char *data, size_t size)
{
    std::ofstream outfile(path_file, std::ios::out | std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Cannot open file for writing.\n";
        return false;
    }

    outfile.write(reinterpret_cast<char *>(data), size * sizeof(unsigned char));
    outfile.close();

    free(data);
    return true;
}
void create_image_BW(int *datas_in, unsigned char *datas_out, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        datas_out[i] = (unsigned char)((datas_in[i] % 2) * 255);
    }
}
void create_image_G(int *datas_in, unsigned char *datas_out, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        // std::cout << "(datas_in[i] % 255) = " << (datas_in[i] % 255) << std::endl;
        datas_out[i] = (datas_in[i] % 255);
    }
}
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

    unsigned char *datas_BW = 0;
    unsigned char *datas_G = 0;
    try
    {
        size_t size = parameter_picture.Get_size_array_2D() * sizeof(unsigned char);

        std::cout << "début malloc datas_BW" << std::endl;
        datas_BW = (unsigned char *)malloc(size);
        std::cout << "fin malloc datas_BW" << std::endl;

        cudaError_t cudaStatus;

        std::cout << "début du RUN Cuda" << std::endl;
        cudaStatus = RUN(parameter_picture, datas_BW, id_cuda, true);
        std::cout << "fin du RUN Cuda" << std::endl;

        if (cudaStatus == cudaSuccess)
        {
            // write_bin(path_bin, datas, parameter_picture.Get_size_array_2D());
            parameter_picture.print_file(path_txt);
            file_generate.exist = true;

            std::cout << "début write_bin_char BW" << std::endl;
            write_bin_char(path_bin + ".BW.bin", datas_BW, parameter_picture.Get_size_array_2D());
            std::cout << "fin write_bin_char BW" << std::endl;
        }

        std::cout << "début malloc datas_BW" << std::endl;
        datas_G = (unsigned char *)malloc(size);
        std::cout << "fin malloc datas_BW" << std::endl;

        std::cout << "début du RUN Cuda" << std::endl;
        cudaStatus = RUN(parameter_picture, datas_G, id_cuda, false);
        std::cout << "fin du RUN Cuda" << std::endl;
        if (cudaStatus == cudaSuccess)
        {

            std::cout << "début write_bin_char G" << std::endl;
            write_bin_char(path_bin + ".G.bin", datas_G, parameter_picture.Get_size_array_2D());
            std::cout << "Fin write_bin_char G" << std::endl;

        }
    }
    catch (const std::exception &)
    {
        free(datas_BW);
        free(datas_G);
        file_generate.exist = false;
        if (if_file_exist(path_txt))
            std::remove(path_txt.c_str());
        if (if_file_exist(path_bin))
            std::remove(path_bin.c_str());
    }

    return file_generate;
}

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

// Fonction pour ouvrir un fichier texte et lire son contenu
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

int main()
{
    // coté en pixel d'une tuile, il y a int(sqrt(lenG)) de tuile par coté
    // exemple pour 720 ==> il y a  int(sqrt(720)) = 26 tuiles donc 26*720 = 18 720 px de coté soit une image de 350 438 400 px en tout
    // donc un fichier binaire en long de 2 803 507 200 octes soit 2.8 Go
    // donc un fichier binaire en int de 1 401 353 600 octes soit 1.4 Go
    const long lenG = 4096; //1792 - 256;

    // nombre de fichier binaire max non traité par le scripte python
    const int max_bin_files = 4;

    // Borne min max de X
    const double coef_x_min = -1.5;
    const double coef_x_max = 1.5;

    // pas d'itération de X et Y
    const double coef_pas = 0.5;

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

    // Construction du nom de base du répertoire
    std::string baseDir = "datas_" + id_cuda_str + "_" + std::to_string(lenG) + "p";
    ParameterPicture parameter_picture(9, lenG, make_double2(-2.0, -1.3), (1.3 + 1.3) / (double)floorf(sqrtf((float)lenG)), 2, (256 * 6), Type_Fractal::Mandelbrot);
    parameter_picture.type_variable = Type_Variable::int_32;
    run(parameter_picture, baseDir, id_cuda);
}
