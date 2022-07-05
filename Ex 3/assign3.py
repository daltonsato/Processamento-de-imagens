'''
Aluno: Dalton Hiroshi Sato
nUSP: 11275172
SCC0251 - Processamento de Imagens
Ano: 2022
Semestre: 1º
Assignment 3 - Filtering in spacial and Frequency Domain
'''

import numpy as np
import imageio
#import matplotlib.pyplot as plt

#def plot_image(digital_image: np.ndarray):
#    '''
#    Auxiliary to plot the image generated.
#    Deactivated to run.codes purposes.
#    '''
#    plt.figure()
#    plt.imshow(digital_image, cmap="gray")
#    plt.show()

def normalize(
    scene_image: np.ndarray) -> np.ndarray:

    '''
    Normalize the image from 0 to 1 (float)
    '''

    scene_image = (scene_image - np.min(scene_image[:]))/(np.max(scene_image[:]) - np.min(scene_image[:]))
    return scene_image


def change_max_value(
    scene_image: np.ndarray,
    max_value: int) -> np.ndarray:

    '''
    Change max values of "pixels" in the image to the max_value given
    '''

    scene_image = scene_image * max_value

    return scene_image


def printRMSE(
    input_image: np.ndarray,
    reference_image: np.ndarray):

    '''
    Prints the RMSE according to the specification.
    '''

    def RMSE(
        input_image: np.ndarray,
        output_image: np.ndarray) -> float:

        '''
        Calculates the diference between the generated image and a comparison image
        '''
        
        input_image = input_image.astype(float)
        output_image = output_image.astype(float)

        error: float = 0

        for i, j in np.ndindex(input_image.shape):
            error += (input_image[i, j] - output_image[i, j])**2

        error /= input_image.shape[0]*input_image.shape[1]

        return np.sqrt(error)

    print("%.4f" % RMSE(input_image, reference_image))


def FFTShift(
    fourierSpec: np.ndarray) -> np.ndarray:

    '''
    Shift the lower frequencies of the given image to the center of the fourier spectrum.
    '''

    M, N = fourierSpec.shape[0:2]

    shiftedSpec: np.ndarray = np.zeros(fourierSpec.shape, dtype = np.complex_)

    x: int = int(N//2)
    y: int = int(M//2)

    shiftedSpec[:x, :y] = fourierSpec[x:, y:]
    shiftedSpec[x:, :y] = fourierSpec[:x, y:]
    shiftedSpec[:x, y:] = fourierSpec[x:, :y]
    shiftedSpec[x:, y:] = fourierSpec[:x, :y]

    return shiftedSpec


def filter_application(
    input_image: np.ndarray,
    filter: np.ndarray) -> np.ndarray:

    '''
    Apply the given filter (image) to the input image, multiplying in the frequency domain.
    Its result is the same as the convolution in the spatial domain.
    '''

    #Create the fouries spectrum
    fourierSpec: np.complex_ = np.fft.fft2(input_image)
    fourierSpec = FFTShift(fourierSpec)

    #Apply the filter
    filter = normalize(filter)
    fourierSpec = np.multiply(fourierSpec, filter)

    #Shift the spectrum, then generate the spacial domain
    fourierSpec = FFTShift(fourierSpec)
    output_image: np.ndarray = np.real(np.fft.ifft2(fourierSpec))

    #Normalize and change the max value of the output generated
    output_image = normalize(output_image)
    output_image = change_max_value(output_image, 255)
    output_image.astype(np.uint8)

    return output_image


def main():
    
    input_image_name: str = input().rstrip()
    filter_name: str = input().rstrip()
    reference_image_name: str = input().rstrip()

    input_image: np.ndarray = imageio.imread(input_image_name)
    filter: np.ndarray = imageio.imread(filter_name)
    reference_image: np.ndarray = imageio.imread(reference_image_name)

    filtered_image: np.ndarray = filter_application(input_image, filter)

    printRMSE(filtered_image, reference_image)
    #plot_image(filtered_image)

if __name__ == '__main__':
    main()