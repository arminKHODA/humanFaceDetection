import os

from . import data
from . import faceDetection
from multiprocessing import Pool
import multiprocessing
from PIL import Image
import dlib
import cv2
import numpy as np
import time
from tqdm import tqdm



def directories_finder(image_so):

    print(f"{data.GREY}directories found : {data.ENDC}")
    directories_in_image_so = [d for d in os.listdir(image_so) if os.path.isdir(os.path.join(image_so, d))]


    if not directories_in_image_so:
        image_folders = [image_so]
    else:    
        image_folders = [os.path.join(image_so, d) for d in directories_in_image_so]


    print ("---- ---- ----")         

    return image_folders

def image_detection(directory, chain=False, humandetection=False, BATCH_SIZE=100):
    print ("\n")  

    print(f"{data.GREY}Processing directory:{data.ENDC} {directory}")
    


    image_files = []


    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')


    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(file)


    image_finder(image_files, directory, chain, humandetection, BATCH_SIZE)

def image_finder(image_list, folder, chain=False, humandetection=False, BATCH_SIZE=100):

    folder_name = os.path.basename(folder).lstrip('0123456789_')
    image_file_name = f"{folder_name}.jpg"
    removed_images = [image_file_name] if image_file_name in image_list else []
    additional_removed_images = [image for image in image_list if os.path.exists(os.path.join(folder, os.path.splitext(image)[0] + '.txt'))]
    removed_images += additional_removed_images
    updated_image_list = [image for image in image_list if image not in removed_images]

    print(f"Found {data.GREEN}{len(image_list)}{data.ENDC} images.")
    print(f"{data.BRIGHT_MAGENTA}{len(removed_images)} {data.ENDC}images will be skipped.")
    print(f"{data.GREEN}{len(updated_image_list)} {data.ENDC}images will be processed.")
    process_image_batch(updated_image_list, folder, chain, humandetection, BATCH_SIZE)


def load_images(image_paths):
    return [Image.open(image_path) for image_path in image_paths]

def process_single_image(args):
    image_path, folder, chain, humandetection = args
    image = Image.open(os.path.join(folder, image_path))
    head_detection(image, chain, humandetection)
    
    image.close()
    return image_path

def process_image_batch(updated_image_list, folder, chain=False, humandetection=False, BATCH_SIZE=100):
    start_time = time.time()
    total_batches = (len(updated_image_list) + BATCH_SIZE - 1) // BATCH_SIZE


    for i in range(total_batches):
        batch_start_time = time.time()
        
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, len(updated_image_list))
        image_batch_paths = updated_image_list[start_index:end_index]
        print(f"Processing batch {data.YELLOW}{i + 1}{data.ENDC}/{data.GREEN}{total_batches}{data.ENDC} with {data.YELLOW}{len(image_batch_paths)}{data.ENDC} images...")


        with Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_image, [(image_path, folder, chain, humandetection) for image_path in image_batch_paths]), total=len(image_batch_paths), desc="Processing images in batch", leave=False))

        batch_end_time = time.time() 
        batch_elapsed_time = batch_end_time - batch_start_time
        formatted_batch_time = format_time(batch_elapsed_time)
        print(f"{data.GREY}Batch {i + 1} processed in {formatted_batch_time}{data.ENDC}")

    end_time = time.time() 
    total_elapsed_time = end_time - start_time
    formatted_total_time = format_time(total_elapsed_time)
    print(f"{data.GREY}All batches processed in {formatted_total_time}{data.ENDC}")

    print(f"{data.GREEN}All batches processed.{data.ENDC}")



def head_detection(image, chain=False, humandetection=False):


    if not chain and not humandetection:
        faces = faceDetection.face_detection_base(image, chain, humandetection)
    elif chain and not humandetection:
        faces = faceDetection.face_detection_exprt(image, humandetection)
    elif chain and humandetection:
        faces = faceDetection.humanFinder(image)
    else:
        faces = None 
    crop(image, faces)



def crop(image, faces, padding_pixels=100):
    image_np = np.array(image)
    img_height, img_width = image_np.shape[:2]
    
    crops = []
    for face in faces:

        top = max(0, face.top() - padding_pixels)
        bottom = min(img_height, face.bottom() + padding_pixels)
        left = max(0, face.left() - padding_pixels)
        right = min(img_width, face.right() + padding_pixels)
        
        cropped_image = image_np[top:bottom, left:right]
        crops.append(cropped_image)
        
        
        padded_face = dlib.rectangle(left, top, right, bottom)
        save_coordinate(cropped_image, image.filename, padded_face)
        
    return crops

def save_coordinate(image, image_path, face):
    img_height, img_width = image.shape[:2]

    heads = [face]

    output_path = os.path.splitext(image_path)[0] + '.txt'
    write_coordinates_to_file(heads, output_path, img_width, img_height)

def write_coordinates_to_file(heads, output_path, img_width, img_height):
    with open(output_path, 'w') as f:
        for head in heads:
            x1 = head.left()
            y1 = head.top()
            x2 = head.right()
            y2 = head.bottom()
            f.write(f"{x1},{y1},{x2},{y2}\n")

def format_time(seconds):
    if seconds >= 60:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)} minutes {seconds:.2f} seconds"
    else:
        return f"{seconds:.2f} seconds"



