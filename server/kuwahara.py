import cv2
#from pykuwahara import kuwahara
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
    filter_applied = custom_kuwahara(image, method=method, radius=radius)  #kuwahara(image, method=method, radius=radius)
    examples_dir = 'examples/'
    if not (os.path.exists(examples_dir)):
        os.mkdir(examples_dir)
    cv2.imwrite(f'./examples/{img_path}_kuwahara_{method}_r{radius}.jpg', filter_applied)
    print(f'successfully applied kuwahara of method: {method} with kernel radius {radius}')


def custom_kuwahara(orig_img, method='mean', radius=3, sigma=None, grayconv=cv2.COLOR_BGR2GRAY, image_2d=None):
    #  Based off of the code from: https://github.com/yoch/pykuwahara/blob/main/src/pykuwahara/kuwahara.py
    if orig_img.ndim != 2 and orig_img.ndim != 3:
        raise TypeError("Incorrect number of dimensions (excepted 2 or 3)")

    if not isinstance(radius, int):
        raise TypeError('`radius` must be int')

    if radius < 1:
        raise ValueError('`radius` must be greater or equal 1')

    if method not in ('mean', 'gaussian'):
        raise NotImplementedError('unsupported method %s' % method)

    if method == 'gaussian' and sigma is None:
        sigma = -1
        # then computed by OpenCV as : 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        # see: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa

        # convert to float32 if necessary for further math computation
    image = orig_img.astype(np.float32, copy=False)

    if image_2d is not None:
        image_2d = image_2d.astype(image.dtype, copy=False)

    # preallocate these arrays
    avgs = np.empty((4, *image.shape), dtype=image.dtype)
    stddevs = np.empty((4, *image.shape[:2]), dtype=image.dtype)

    if image.ndim == 3:
        if image_2d is None:
            # NOTE this doesn't support float64
            image_2d = cv2.cvtColor(orig_img, grayconv).astype(image.dtype, copy=False)
        avgs_2d = np.empty((4, *image.shape[:2]), dtype=image.dtype)
    elif image.ndim == 2:
        image_2d = image
        avgs_2d = avgs

    # Create a pixel-by-pixel square of the image
    squared_img = image_2d ** 2

    if method == 'mean':
        kxy = np.ones(radius + 1, dtype=image.dtype) / (radius + 1)  # kernelX and kernelY (same)
    elif method == 'gaussian':
        kxy = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=cv2.CV_32F) #ADD Lagrange Function
        kxy /= kxy[radius:].sum()  # normalize the semi-kernels
        klr = np.array([kxy[:radius + 1], kxy[radius:]])
        kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]

    # the pixel position for all kernel quadrants
    shift = [(0, 0), (0, radius), (radius, 0), (radius, radius)]

    # Calculation of averages and variances on subwindows
    for k in range(4):
        if method == 'mean':
            kx = ky = kxy
        elif method == 'gaussian':
            kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image, -1, kx, ky, avgs[k], shift[k])
        if image.ndim == 3:  # else, this is already done...
            cv2.sepFilter2D(image_2d, -1, kx, ky, avgs_2d[k], shift[k])
        cv2.sepFilter2D(squared_img, -1, kx, ky, stddevs[k], shift[k])
        stddevs[k] = stddevs[k] - avgs_2d[k] ** 2  # compute the final variance on subwindow

    # Choice of index with minimum variance
    indices = np.argmin(stddevs, axis=0)

    # Building the filtered image
    if image.ndim == 2:
        filtered = np.take_along_axis(avgs, indices[None, ...], 0).reshape(image.shape)
    else:  # then avgs.ndim == 4
        filtered = np.take_along_axis(avgs, indices[None, ..., None], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)


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