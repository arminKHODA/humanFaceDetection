import time
import os
import cv2
import concurrent.futures
import dlib
from PIL import Image


from lib import data
from lib import function

if __name__ == '__main__':
    
    input(f"press {data.BRIGHT_BLUE}Enter{data.ENDC} to start")
    

    default_image_so = r'C:\Desktop'
    
    image_so = input('input root path for face detection, Enter for default path:')

    if not image_so.strip():
        image_so = default_image_so
    print(f"{data.GREY}image source folder:", image_so)
    print(f"{data.ENDC}")


    print(f"{data.GREY}Select detection mode:{data.ENDC}")
    print(f"1{data.GREY}. simple{data.ENDC}")
    print(f"2{data.GREY}. advance{data.ENDC}")
    print(f"3{data.GREY}. expert{data.ENDC}")
    mode_selection = input(f"{data.GREY}Enter 1, 2, or 3: {data.ENDC}")
    print("\n")

    chain = False
    humandetection = False
    BATCH_SIZE=100
    
    if mode_selection == '2':
        chain = True
        BATCH_SIZE=5
    elif mode_selection == '3':
        chain = True
        humandetection = True
        BATCH_SIZE=2
    

    image_folders = function.directories_finder(image_so)
    for folder in image_folders:
        print(folder)
    print("---- ---- ----\n")
    for folder in image_folders:
        function.image_detection(folder, chain, humandetection, BATCH_SIZE)
        
    
    print("\n")
    print("Done")
    input('Enter to exit')
