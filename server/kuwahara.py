import cv2
from pykuwahara import kuwahara
from typing import Literal
import numpy as np

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
    cv2.imwrite(f'./examples/{img_path}_kuwahara_{method}_r{radius}.jpg', filter_applied)
    print(f'successfully applied kuwahara of method: {method} with kernel radius {radius}')

if __name__ == '__main__':
    # various radius - mean
    library_kuwahara('./me-lol.jpg', radius=1)
    library_kuwahara('./me-lol.jpg', radius=2)
    library_kuwahara('./me-lol.jpg', radius=5)
    library_kuwahara('./me-lol.jpg', radius=10)
    library_kuwahara('./me-lol.jpg', radius=20)

    # various radius - gaussian
    library_kuwahara('./me-lol.jpg', method='gaussian', radius=1)
    library_kuwahara('./me-lol.jpg', method='gaussian', radius=2)
    library_kuwahara('./me-lol.jpg', method='gaussian', radius=5)
    library_kuwahara('./me-lol.jpg', method='gaussian', radius=10)
    library_kuwahara('./me-lol.jpg', method='gaussian', radius=20)
    library_kuwahara('./me-lol.jpg', method='gaussian', radius=50)
