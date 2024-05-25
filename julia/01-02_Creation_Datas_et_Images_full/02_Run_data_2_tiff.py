####################  1. Les libraires  ####################

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

####################  2. Les constantes ####################

# Nombre de Thread lancé
NB_THREAD = 4
#Noms des fichiers constant
FILE_TXT = "parameters.txt"
FILE_BIN = "data.bin"
FILE_ZIP = "data.zip"
FILE_7ZIP = "data.7z"
FILE_NB_TIF= "out1.tif"
FILE_G_TIF= "out2.tif"

####################  3. La class ParameterPicture et son lecteur  ####################

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
    type_variable=0
# Lecteur de la class ParameterPicture
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

####################  4. les fonctions cuda  :  ####################

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


def save_tif(image, filename):
    # Sauvegarder l'image en TIFF avec compression
    image.save(filename, format='TIFF', compression='tiff_lzw')


def lister_paths_bin(baseDir:str,nb_thread:int,no_thread:int):
    liste_paths= list()
    for path, subdirs, files in os.walk(baseDir):
        for name in files:
            if name==FILE_BIN:
                num = int(path.split("id_")[-1])
                if num % nb_thread == no_thread:
                    liste_paths.append(path)
    return liste_paths

def bin_file2image_np_2D(file_bin:str,param:ParameterPicture):
    data = numpy.zeros(0)
    start_l = time.time()
    with open(file_bin, 'rb') as f:
        if param.type_variable == 32 :
            data = numpy.fromfile(f, dtype=numpy.int32)
        elif  param.type_variable == 64 :
            data = numpy.fromfile(f, dtype=numpy.int64)
        else:
            raise Exception(f"type de variable != 32 et 64 => {param.type_variable}")
    elapsed = time.time() - start_l
    print(f'Temps d\'execution  Open : {elapsed:.4} s')

    start_l = time.time()
    data = data.reshape(param.lenG*param.lenL,param.lenG*param.lenL)
    elapsed = time.time() - start_l
    print(f'Temps d\'execution  1d --> 2D : {elapsed:.4} s')
    return data

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


def main_bin_2_tif():
    f = open("parameters/id_cuda.txt","r")
    id_cuda = int(f.readline())
    f.close()
    print(f" id cuda = {id_cuda}")
    cuda.select_device(id_cuda)

    nb_thread = NB_THREAD
    values = range(0,nb_thread)
    Parallel(n_jobs=nb_thread,prefer="threads")(delayed(sub_main_bin_2_tif)(value) for value in values)


if __name__ == "__main__":
    start_g = time.time()
    main_bin_2_tif()
    end_g = time.time()
    elapsed = end_g - start_g
    print(f'Temps d\'execution  G: {elapsed} s')
