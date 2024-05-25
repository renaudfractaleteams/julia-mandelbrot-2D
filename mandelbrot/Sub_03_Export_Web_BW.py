import os
import shutil
from math import *
from pathlib import Path
import lib_deepzoom as deepzoom
from joblib import Parallel, delayed

# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
NB_THREAD = 1
FILE_TXT = "parameters.txt"
FILE_NB_TIF= "out1.tif"
FILE_G_TIF= "out2.tif"
FOLDER_EXPORT_TIF = "./web/download"
FOLDER_EXPORT_DZI = "./web/pan"

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

def create_dzi(file_tif:str,path_export:str):
    creator = deepzoom.ImageCreator(
    tile_size=1024,
    tile_overlap=2,
    tile_format="png",
    image_quality=0.8,
    resize_filter="bicubic",
    )
    name = file_tif.split("/")[-1].replace(".tif","")
    path_dzi = path_export+"/"+name+".dzi"
    creator.create(file_tif, path_dzi)
    return path_dzi


def lister_paths_file_type(baseDir:str,nb_thread:int,value:int, file_type:str):
    liste_paths= list()
    for path, subdirs, files in os.walk(baseDir):
        for name in files:
            if name==file_type:
                num = int(path.split("id_")[-1])
                if num % nb_thread == value:
                    liste_paths.append(path)
    return liste_paths


def sub_main(value:int,file_type:str):
    nb_thread = NB_THREAD
    baseDir = "."
    liste_dir = os.listdir(baseDir)
    liste_paths = list()
    for dir_name  in liste_dir:
        if dir_name.startswith("datas_0_4096p"):
            liste_paths += lister_paths_file_type(baseDir,nb_thread,value,file_type)
    for path_dir in liste_paths:
        file_txt = os.path.join(path_dir, FILE_TXT)
        file_tif = os.path.join(path_dir, file_type)

        if os.path.isfile(file_txt) and os.path.isfile(file_tif):
            id= int(path_dir.split("id_")[-1])
            nbp = path_dir.split("/")[-2].split("_")[-1]
            param = generate_ParameterPicture(file_txt)
            print(path_dir[2:].split("/")[0])
            x_coef = int(param.coef_julia_x*100)
            y_coef = int(param.coef_julia_y*100)
            dir_export =FOLDER_EXPORT_DZI

            file_sans_ext = file_type.split(".")[0]
            file_dzi_tif = f"{file_sans_ext}_{nbp}_{id}.tif"
            new_file_tif = os.path.join(path_dir,file_dzi_tif)
            shutil.copyfile(file_tif,new_file_tif)

            print(dir_export)
            if not os.path.isdir(dir_export):
                print(f"création dir export : {dir_export}")
                Path(dir_export).mkdir(parents=True,exist_ok=True)
            path_dzi =""
            try:
                path_dzi = create_dzi(file_tif=new_file_tif,path_export=dir_export)
            except Exception as e:
                print(str(e))
            
            if path_dzi!= "" and os.path.isfile(new_file_tif):
                os.remove(file_tif)
                file_export = os.path.join(FOLDER_EXPORT_TIF,file_dzi_tif)
                shutil.copyfile(new_file_tif,file_export)

def main_run(file_type:str):
    nb_thread = NB_THREAD
    values = range(0,nb_thread)
    Parallel(n_jobs=nb_thread,prefer="threads")(delayed(sub_main)(value, file_type) for value in values)

def main_03_export_web():
    if not os.path.isdir(FOLDER_EXPORT_DZI):
        Path(FOLDER_EXPORT_DZI).mkdir(parents=True,exist_ok=True)
    if not os.path.isdir(FOLDER_EXPORT_TIF):
        Path(FOLDER_EXPORT_TIF).mkdir(parents=True,exist_ok=True)
    main_run(FILE_NB_TIF)
    #main_run(FILE_G_TIF)

if __name__ == "__main__":
    main_03_export_web()
