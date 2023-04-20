import cv2
from pykuwahara import kuwahara
from typing import Literal
import numpy as np
import os

def library_kuwahara(img_path: str, method: Literal['mean', 'gaussian']='mean', radius: int=3) -> None:
    """
    
    This method simply calls the kuwahara function from the pykuwahara library with the given method and radius
    Should be used for testing our approximation kuwahara method
    Files are output to ./examples/

    Args:
        img_path (str): The path to the image
        method (Literal['mean', 'gaussian], optional): The kernel method, defaults to 'mean'.
        radius (int, optional): The radius of the kernel, defaults to 3.
    """
    image = cv2.imread(img_path)
    filter_applied = kuwahara(image, method=method, radius=radius)
    examples_dir = 'examples/'
    if not (os.path.exists(examples_dir)):
        os.mkdir(examples_dir)
    cv2.imwrite(f'./examples/{img_path}_kuwahara_{method}_r{radius}.jpg', filter_applied)
    print(f'successfully applied kuwahara of method: {method} with kernel radius {radius}')

if __name__ == '__main__':
    example_imgs = ['me-lol.jpg', 'turing-test.jpg', 'cherry-blossoms.JPG', 'grassy-field.JPG']

    for img_path in example_imgs:
        # various radius - mean
        library_kuwahara(img_path)
        library_kuwahara(img_path, radius=5)
        library_kuwahara(img_path, radius=10)
        library_kuwahara(img_path, radius=20)
        library_kuwahara(img_path, radius=50)
        library_kuwahara(img_path, radius=100)

        # various radius - gaussian
        library_kuwahara(img_path, method='gaussian')
        library_kuwahara(img_path, method='gaussian', radius=5)
        library_kuwahara(img_path, method='gaussian', radius=10)
        library_kuwahara(img_path, method='gaussian', radius=20)
        library_kuwahara(img_path, method='gaussian', radius=50)
        library_kuwahara(img_path, method='gaussian', radius=100)