import os, json
import shutil
from pprint import pprint
FOLDER_EXPORT_TIF_ANALYSE = "./mandelbrot/download"
FOLDER_EXPORT_DZI_ANALYSE = "./mandelbrot/pan"

FOLDER_EXPORT_TIF_USE = "/download"
FOLDER_EXPORT_DZI_USE = "/pan"

def lister_paths_file_tif():
    liste_path_files= list()
    dir_base  = FOLDER_EXPORT_TIF_ANALYSE
    for path, subdirs, files in os.walk(dir_base):
        for name in files:
            if name.split(".")[-1]=="tif":
                path_file = os.path.join(path,name)
                liste_path_files.append(path_file)
    return liste_path_files

def lister_paths_file_dzi():
    liste_path_files= list()
    dir_base  = FOLDER_EXPORT_DZI_ANALYSE
    for path, subdirs, files in os.walk(dir_base):
        for name in files:
            if name.split(".")[-1]=="dzi":
                path_file = os.path.join(FOLDER_EXPORT_DZI_USE,name)
                path_file_OK=path_file.replace("\\","/")
                liste_path_files.append(path_file_OK)
    return liste_path_files

def index_dzi():
    files_path = lister_paths_file_dzi()
    files_path.sort()
    pprint(files_path)
    dico = dict()
    min_x= None
    max_x= None
    min_y= None
    max_y= None
    for file_path in files_path:
        name_file_tif = file_path.split("/")[-1].replace(".dzi",".tif")
        print(name_file_tif)
        path_file_tif  = FOLDER_EXPORT_TIF_ANALYSE+"/"+name_file_tif
        if os.path.isfile(path_file_tif):
            dico[file_path]=FOLDER_EXPORT_TIF_USE+"/"+name_file_tif
        else:
            dico[file_path]=""
    str_js = "dataBase = " + json.dumps(dico,indent=4)+"\n"
    str_js += "min_x = "+str(min_x)+"\n"
    str_js += "max_x = "+str(max_x)+"\n"
    str_js += "min_y = "+str(min_y)+"\n"
    str_js += "max_y = "+str(max_y)+"\n"

    file_object  = open("./mandelbrot/js/data.js", "w")
    file_object.write(str_js)
    file_object.close()

if __name__ == "__main__":
    index_dzi()