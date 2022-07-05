'''
Aluno: Dalton Hiroshi Sato
nUSP: 11275172
SCC0251 - Processamento de Imagens
Ano: 2022
Semestre: 1º
Assignment 4 - Image Restoration
'''

# Imports section
#from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import numpy as np
import matplotlib.pyplot as plt
import imageio
#import cv2 

##################################################################################################
# Auxiliary functions

def normalize(
    scene_image: np.ndarray) -> np.ndarray:

    '''
    Normalize the image from 0 to 1 (float)

    Parameters:
    scene_image(np.ndarray): Given image

    Return:
    np.ndarray: Normalized image
    '''
    scene_image.astype(np.float32)
    scene_image = (scene_image - np.min(scene_image[:]))/(np.max(scene_image[:]) - np.min(scene_image[:]))

    return scene_image


def change_max_value(
    scene_image: np.ndarray,
    max_value: int) -> np.ndarray:

    '''
    Change max values of "pixels" in the image to the max_value given

    Parameters:
    scene_image (np.ndarray): Given image
    max_value (int): Maximum value to be applied, proportionaly, to the image

    Return:
    np.ndarray: Image with a new max value
    '''

    scene_image = scene_image * max_value

    return scene_image


def printRMSE(
    input_image: np.ndarray,
    reference_image: np.ndarray) -> None:

    '''
    Prints the RMSE according to the specification.

    Parameters:
    input_image (np.ndarray): Original image
    reference_image (np.ndarray): Altered image

    Return:
    None
    '''

    def RMSE(
        input_image: np.ndarray,
        output_image: np.ndarray) -> float:

        '''
        Calculates the diference between the generated image and a comparison image.
        
        Parameters:
        input_image (np.ndarray): Original image
        output_image (np.ndarray): Altered image

        Return:
        float: Error calculated
        '''
        
        input_image = input_image.astype(float)
        output_image = output_image.astype(float)

        error: float = 0

        for i, j in np.ndindex(input_image.shape):
            error += (input_image[i, j] - output_image[i, j])**2

        error /= input_image.shape[0]*input_image.shape[1]

        return np.sqrt(error)

    print("%.4f" % RMSE(input_image, reference_image))


def get_motion_psf(
    dim_x: int, dim_y: int, degree_angle: float, num_pixel_dist: int = 20)-> np.ndarray:
    """Essa função retorna uma array representando a PSF para um dado ângulo em graus

    Parameters:
    -----------
        dim_x: int
            The width of the image.
        dim_y: int
            The height of the image.
        degree_angle: float
            The angle of the motion blur. Should be in degrees. [0, 360)
        num_pixel_dist: int
            The distance of the motion blur. [0, \infinity).
            Remember that the distance is measured in pixels.
            Greater will be more blurry.
    
    Returns:
    --------
        np.ndarray
            The point-spread array associated with the motion blur.

    """
    psf = np.zeros((dim_x, dim_y))
    center = np.array([dim_x-1, dim_y-1])//2
    radians = degree_angle/180*np.pi
    phase = np.array([np.cos(radians), np.sin(radians)])
    for i in range(num_pixel_dist):
        offset_x = int(center[0] - np.round_(i*phase[0]))
        offset_y = int(center[1] - np.round_(i*phase[1]))
        psf[offset_x, offset_y] = 1 
    psf /= psf.sum()
 
    return psf 

def gaussian_noise(
    size, mean=0, std=0.01):
    '''
    Generates a matrix with Gaussian noise in the range [0-255] to be added to an image
    
    :param size: tuple defining the size of the noise matrix 
    :param mean: mean of the Gaussian distribution
    :param std: standard deviation of the Gaussian distribution, default 0.01
    :return matrix with Gaussian noise to be added to image
    '''
    noise = np.multiply(np.random.normal(mean, std, size), 255)
    
    return noise

def gaussian_filter(
    k=5, sigma=1.0) -> np.ndarray:

    ''' Gaussian filter
    :param k: defines the lateral size of the kernel/filter, default 5
    :param sigma: standard deviation (dispersion) of the Gaussian distribution
    :return matrix with a filter [k x k] to be used in convolution operations
    '''
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)


def myConvolve2d(
    original_image: np.ndarray, 
    kernel: np.ndarray) -> np.ndarray:
    '''
    Convolution 2D of a image, given a kernel to apply, similar to the process done in the assigment 2

    Parameters:
    original_image (np.ndarray): Given image to convolve
    kernel (np.ndarray): Kernel applied in the convolution
        
    Returns:
    np.ndarray: Convolved image
    '''

    M, N = original_image.shape

    filter_size: int = kernel.shape[0]
    ratio: int = int(filter_size/2)

    padded_image = np.pad(original_image, ratio, 'edge')
    convolvedImage = np.zeros((M,N), np.float32)

    for x, y in np.ndindex (M, N):
        convolvedImage[x, y] = np.sum(padded_image[x:x+filter_size, y:y+filter_size] * kernel[:filter_size, :filter_size])
            
    convolvedImage.astype(np.uint8)

    return convolvedImage


##################################################################################################
# Main functions

def method_one(
    original_image: np.ndarray) -> np.ndarray:
    '''
    Method that will apply the CLSQ (Constraint least-squares Filter) to the image.

    Parameters:
    original_image (np.ndarray): Given image to restore

    Return:
    np.ndarray: Restored image
    '''

    def constrained_ls_filter(
        blurred_image: np.ndarray, 
        kernel: np.ndarray, 
        laplacian: np.ndarray, 
        gamma: float) -> np.ndarray:
        '''
        CLSQ filter.

        Parameters:
        blurred_image (np.ndarray): Image blurred to be restored
        kernel (np.ndarray): PSF generated from the image
        laplacian (np.ndarray): Laplacian filter
        gamma (float): Gamma value for the CLSQ image

        Return:
        np.ndarray: Restored image
        '''
        restored_image = (blurred_image)
        restored_image = fftshift(fft2(restored_image))
        kernel = fftshift(fft2(kernel, s=blurred_image.shape))

        P = fftshift(fft2(laplacian, s=blurred_image.shape))

        # Apply the CLSQ filter, as shown in the class
        kernel = np.conj(kernel) / ((np.abs(kernel)**2) + (gamma * np.abs(P)**2))

        restored_image = restored_image * kernel
        restored_image = np.abs(ifft2(ifftshift(restored_image)))
        return restored_image

    degradation_filter_size: int = int(input())
    sigma_value: float = float(input())
    gamma_value: float = float(input())

    blurred_image: np.ndarray = (original_image)

    # Here we blur the image so we can 'unblur' it
    g_filter: np.ndarray = gaussian_filter(degradation_filter_size, sigma_value)

    blurred_image = myConvolve2d(original_image, g_filter)

    # Laplacian operator
    P = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    blurred_image = constrained_ls_filter(blurred_image, g_filter, P, gamma_value)
    blurred_image = np.clip(blurred_image.astype(int), 0, 255)

    return blurred_image.astype(np.uint8)


def method_two(
    original_image: np.ndarray) -> np.ndarray:

    '''
    Method that will apply the Richardson-Lucy deconvolution algorithm.

    Parameters:
    original_image (np.ndarray): Given image

    Return:
    np.ndarray: Restored image after the RL algorithm.
    '''

    def richardson_lucy_filter(
        blurred_image: np.ndarray, 
        psf: np.ndarray, 
        max_iter: int=50) -> np.ndarray:
        """
        Richardson-Lucy filter.

        Parameters:
        ----------
            blurred_image : np.ndarray
                Your degraded image (grayscale)
            psf : np.ndarray
            max_iter : int
                maximum number of iterations
        Returns
        -------
            np.ndarray
        """
        
        # Following the algorithm in https://www.strollswithmydog.com/richardson-lucy-algorithm/ , we make all the convolution in the frequency domain, avoiding true division by 0
        Ok = myConvolve2d((blurred_image), gaussian_filter() )
        for _ in range(max_iter):
            Ok_fou: np.ndarray = fftshift(fft2(Ok))

            # Calculates the denominator of the formula
            idenom: np.ndarray = np.multiply(Ok_fou, psf)
            denom: np.ndarray = ifft2(ifftshift(idenom))

            # Calculates the ratio
            ratio: np.ndarray = blurred_image/denom
            iratio: np.ndarray = fftshift(fft2(ratio))

            # Calculates the product
            iproduct: np.ndarray = np.multiply(iratio, np.flip(psf))
            product: np.ndarray = ifft2(ifftshift(iproduct))

            # Finds the iteration image, using in the next iteration
            Ok = product * Ok
        
        # Return the real value - ignore the complex space
        return np.abs(Ok)


    angle: float = float(input())
    steps: int = int(input())
    pixel_distance: int = 20

    restored_image: np.ndarray = (original_image)

    M, N = original_image.shape
    kernel: np.ndarray = np.ones((5, 5), np.float32)/25
    psf: np.ndarray = get_motion_psf(M, N, angle, pixel_distance)

    # Calculates the new psf, as shown in the colab 4
    new_psf: np.ndarray = myConvolve2d(psf, kernel)

    # Filter apply
    restored_image = richardson_lucy_filter(original_image, new_psf, steps)

    # Simple normalization
    restored_image = normalize(restored_image)
    restored_image = change_max_value(restored_image, 255)
    restored_image.astype(np.uint8)

    return restored_image


##################################################################################################
# Main

def main():
    '''
    Main function. Given an input image and a selected choice (1 for CLSQ and 2 for RL deconvolution),
    it will be restored where possible following the steps instructed.
    '''
    
    input_image_name: str = input().rstrip()
    method_choice: int = int(input())
    
    original_image: np.ndarray = imageio.imread(input_image_name)
    restored_image: np.ndarray

    if(method_choice == 1):
        restored_image = method_one(original_image)
    elif(method_choice == 2):
        restored_image = method_two(original_image)
    else:
        raise ValueError("Valor digitado incorreto!")

    #imageio.imwrite(f'restored_image.jpg', restored_image)
    printRMSE(original_image, restored_image)
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_image, cmap="gray")
    ax2 = fig.add_subplot(122)
    ax2.imshow(restored_image, cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()