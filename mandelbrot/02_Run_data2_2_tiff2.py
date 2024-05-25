####################  1. Les libraires  ####################

# Librairie de gestion de tableau très rapide
import numpy 
# Librairie de gestion de création de fonction cuda directement en python
#from numba import jit, cuda
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
from pprint import pprint
####################  2. Les constantes ####################

# Nombre de Thread lancé
NB_THREAD = 1
#Noms des fichiers constant
FILE_TXT = "parameters.txt"
FILE_BIN_BW = "data.bin.BW.bin"
FILE_BIN_G= "data.bin.G.bin"
FILE_7ZIP_BW = "data_BW.7z"
FILE_7ZIP_G = "data_G.7z"
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

def save_tif(image, filename):
    # Sauvegarder l'image en TIFF avec compression
    image.save(filename, format='TIFF', compression='tiff_lzw')


def lister_paths_bin_g(baseDir:str,nb_thread:int,no_thread:int):
    liste_paths= list()
    for path, subdirs, files in os.walk(baseDir):
        for name in files:
            if name==FILE_BIN_G:
                num = int(path.split("id_")[-1])
                if num % nb_thread == no_thread:
                    liste_paths.append(path)
    return liste_paths

def lister_paths_bin_bw(baseDir:str,nb_thread:int,no_thread:int):
    liste_paths= list()
    for path, subdirs, files in os.walk(baseDir):
        for name in files:
            if name==FILE_BIN_BW:
                num = int(path.split("id_")[-1])
                if num % nb_thread == no_thread:
                    liste_paths.append(path)
    return liste_paths


def bin_file2image_np_2D_char(file_bin:str,param:ParameterPicture):
    data = numpy.zeros(0)
    start_l = time.time()
    with open(file_bin, 'rb') as f:
        data = numpy.fromfile(f, dtype=numpy.uint8)
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
            liste_paths = lister_paths_bin_g(baseDir,nb_thread,value)
            liste_paths+= lister_paths_bin_bw(baseDir,nb_thread,value)
            pprint(liste_paths)
            for path_bin in liste_paths:
                file_txt = os.path.join(path_bin, FILE_TXT)
                file_bin_g = os.path.join(path_bin, FILE_BIN_G)
                file_bin_bw = os.path.join(path_bin, FILE_BIN_BW)
                file_7zip_BW = os.path.join(path_bin, FILE_7ZIP_BW)
                file_7zip_G = os.path.join(path_bin, FILE_7ZIP_G)
                file_nb_tif = os.path.join(path_bin, FILE_NB_TIF)
                file_g_tif = os.path.join(path_bin, FILE_G_TIF)
                
                if os.path.isfile(file_txt) and (os.path.isfile(file_bin_bw) or os.path.isfile(file_bin_g)):
                    start_l = time.time()
                    ########################### generate_ParameterPicture ######################
                    start = time.time()
                    print(f"Use : {file_txt}")
                    param = generate_ParameterPicture(file_txt)
                    end = time.time()
                    elapsed = end - start
                    print(f'Temps d\'execution  generate_ParameterPicture: {elapsed:.4} s')
                    
                    ########################### bin_file2image_np_2D ######################
                    if os.path.isfile(file_bin_bw) :
                        start = time.time()
                        image_raw = bin_file2image_np_2D_char(file_bin_bw,param)
                        end = time.time()
                        elapsed = end - start
                        print(f'Temps d\'execution  bin_file2image_np: {elapsed:.4} s')
                        gc.collect()
                        
                        start = time.time()
                        im_out = Image.fromarray(image_raw).convert('L')
                        save_tif(im_out,file_nb_tif)
                        end = time.time()
                        elapsed = end - start
                        print(f'Temps d\'execution  save tif NB: {elapsed:.4} s')
                        
                        del image_raw
                        del im_out
                        gc.collect()


                        print("Create a 7ZipFile Object")
                        start =  time.time()
                        with py7zr.SevenZipFile(file_7zip_BW, 'w') as z:
                            z.writeall(file_bin_bw)
                        elapsed = time.time() - start
                        print(f'Temps d\'execution  7zip NB : {elapsed:.5} s')
    
                        # Check to see if the zip file is created
                        if os.path.exists(file_7zip_BW):
                            print("7ZIP file created")
                            os.remove(file_bin_bw)
                        else:
                            print("7ZIP file not created")
                        elapsed_l = time.time() - start_l
                    ########################### create img N&B  % 2  ######################
                    if os.path.isfile(file_bin_g) :
                        start = time.time()
                        image_raw = bin_file2image_np_2D_char(file_bin_g,param)
                        end = time.time()
                        elapsed = end - start
                        print(f'Temps d\'execution  bin_file2image_np: {elapsed:.4} s')
                        gc.collect()
                        
                        start = time.time()
                        im_out = Image.fromarray(image_raw).convert('L')
                        save_tif(im_out,file_g_tif)
                        end = time.time()
                        elapsed = end - start
                        print(f'Temps d\'execution  save tif NB: {elapsed:.4} s')
                        
                        del image_raw
                        del im_out
                        gc.collect()


                        print("Create a 7ZipFile Object")
                        start =  time.time()
                        with py7zr.SevenZipFile(file_7zip_G, 'w') as z:
                            z.writeall(file_bin_g)
                        elapsed = time.time() - start
                        print(f'Temps d\'execution  7zip G : {elapsed:.5} s')
    
                        # Check to see if the zip file is created
                        if os.path.exists(file_7zip_G):
                            print("7ZIP file created")
                            os.remove(file_bin_g)
                        else:
                            print("7ZIP file not created")
                        
                    elapsed_l = time.time() - start_l
                    print("***************************************************************************")
                    print(f'Temps d\'execution  {path_bin}: {elapsed_l:.5} s')
                    print("***************************************************************************")
            break

def main_bin_2_tif():
    f = open("parameters/id_cuda.txt","r")
    id_cuda = int(f.readline())
    f.close()
    print(f" id cuda = {id_cuda}")
    #cuda.select_device(id_cuda)

    nb_thread = NB_THREAD
    values = range(0,nb_thread)
    Parallel(n_jobs=nb_thread,prefer="threads")(delayed(sub_main_bin_2_tif)(value) for value in values)


if __name__ == "__main__":
    start_g = time.time()
    main_bin_2_tif()
    end_g = time.time()
    elapsed = end_g - start_g
    print(f'Temps d\'execution  G: {elapsed} s')
