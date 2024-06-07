# Face Detection Package

This package includes a set of Python scripts for detecting and extracting faces from images. The package contains the following scripts:

1. `FaceDetection.py`
2. `faceDetection.py`
3. `function.py`
4. `data.py`

## Features

- Load images from a specified directory.
- Detect faces in the loaded images.
- Save the detected faces to an output directory.

## Requirements

- Python 3.x
- OpenCV (`cv2`)

## Installation

1. Ensure you have Python 3 installed. If not, download and install it from [python.org](https://www.python.org/).
2. Install the OpenCV library using pip:
    ```bash
    pip install opencv-python
    ```

## Usage

### Running the Main Script

1. Place your images in a directory (e.g., `images/`).
2. Run the `FaceDetection.py` script:
    ```bash
    python FaceDetection.py
    ```
3. The detected faces will be saved in the `output/` directory.

### Main Functionalities

1. **Loading Data (data.py)**:
    - The `load_data` function loads all `.jpg` images from the specified directory.

    ```python
    def load_data(directory):
        return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]
    ```

2. **Face Detection (faceDetection.py)**:
    - The `detect_faces` function detects faces in the images using OpenCV's Haar Cascade classifier.

    ```python
    import cv2

    def detect_faces(images):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = []

        for image_path in images:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in detected_faces:
                face = img[y:y+h, x:x+w]
                faces.append((image_path, face))
        
        return faces
    ```

3. **Saving Faces (function.py)**:
    - The `save_faces` function saves the detected faces to the specified output directory.

    ```python
    import os
    import cv2

    def save_faces(faces, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, (image_path, face) in enumerate(faces):
            output_path = os.path.join(output_dir, f'face_{i}.jpg')
            cv2.imwrite(output_path, face)
            print(f"Saved face {i} from {image_path} to {output_path}")
    ```

4. **Main Script (FaceDetection.py)**:
    - The `FaceDetection.py` script integrates the functionalities of loading data, detecting faces, and saving faces.

    ```python
    from faceDetection import detect_faces
    from data import load_data
    from function import save_faces

    def main():
        # Load the data
        images = load_data('images/')
        
        # Detect faces in the images
        faces = detect_faces(images)
        
        # Save the detected faces
        save_faces(faces, 'output/')

    if __name__ == "__main__":
        main()
    ```

## License

This project is licensed under the MIT License.
