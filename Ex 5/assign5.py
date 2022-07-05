'''
Aluno: Dalton Hiroshi Sato
nUSP: 11275172
SCC0251 - Processamento de Imagens
Ano: 2022
Semestre: 1º
Assignment 5 - Morphology and Image Description
'''

# Imports section
import time
#from numpy.fft import fft2, ifft2, fftshift, ifftshift

import numpy as np
#import matplotlib.pyplot as plt
import imageio

##################################################################################################
# Auxiliary functions


def normalize(
        img: np.ndarray) -> np.ndarray:
    '''
    Normalize the image from 0 to 1 (float)

    Parameters:
    img(np.ndarray): Given image

    Return:
    np.ndarray: Normalized image
    '''
    img.astype(np.float32)
    img -= np.min(img)
    img /= np.max(img)

    return img


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
        dim_x: int, dim_y: int, degree_angle: float, num_pixel_dist: int = 20) -> np.ndarray:
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
    convolvedImage = np.zeros((M, N), np.float32)

    for x, y in np.ndindex(M, N):
        convolvedImage[x, y] = np.sum(
            padded_image[x:x+filter_size, y:y+filter_size] * kernel[:filter_size, :filter_size])

    convolvedImage.astype(np.uint8)

    return convolvedImage


##################################################################################################
# Main functions

def RGBtoGrayScale(
        RGBImg: np.ndarray) -> np.ndarray:
    '''
    Auxiliary function that converts a RGB image to its grayScale level,
    using the Luminance Weights.

    Parameters:
    RGBImg (np.ndarray): Input RBG image

    Return:
    np.ndarray: Output image in gray scale
    '''

    R = RGBImg[:, :, 0]
    G = RGBImg[:, :, 1]
    B = RGBImg[:, :, 2]

    outGrayImg = R*0.299 + G*0.587 + B*0.114
    outGrayImg = outGrayImg.astype(np.uint8)
    return outGrayImg


def binarizeImg(
        inputImg: np.ndarray,
        threshold: int) -> np.ndarray:
    '''
    Binarize the given image, using a defined threshold.

    Parameters:
    inputImg (np.ndarray): Given input image
    thresholg (int): Threshold to apply to the image

    Return:
    np.ndarray: Binarized image
    '''

    outputImg: np.ndarray = np.copy(inputImg)
    outputImg[inputImg < threshold] = 0
    outputImg[inputImg >= threshold] = 1

    #outputImg = outputImg.astype(np.uint8)

    return outputImg

def erosion(
        img: np.ndarray) -> np.ndarray:
    '''
    Apply erosion to the given image.

    Parameters:
    inputImg (np.ndarray): Given input image

    Return:
    np.ndarray: Eroded image
    '''

#     # Constrói a estrutura
#     structuring_kernel = np.ones((3, 3))

#     # Faz um pad na imagem
#     orig_shape = img.shape
#     image_pad = np.pad(array=img, pad_width=1, mode='constant')
#     h_reduce, w_reduce = (image_pad.shape[0] - orig_shape[0]), (image_pad.shape[1] - orig_shape[1])

#     # Calcula as submatrizes que conterão a estrutura 3x3
#     flat_submatrices = np.array([
#         image_pad[i:(i + 3), j:(j + 3)]
#         for i in range(image_pad.shape[0] - h_reduce) for j in range(image_pad.shape[1] - w_reduce)
#     ])

#    # Substitui valores com 1 ou 0, dependendo se a estrutura "entra" na sub matriz
#     image_dilate = np.array([1 if (i == structuring_kernel).all() else 0 for i in flat_submatrices])

#     # Reshape para as dimensões originais
#     image_dilate = image_dilate.reshape(orig_shape)

#     return image_dilate

    outImg: np.ndarray = np.zeros(img.shape)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            outImg[i, j] = np.min(img[i-1:i+2, j-1:j+2])

    return outImg

def dilation(
        img: np.ndarray) -> np.ndarray:
    '''
    Apply dilation to the given image.

    Parameters:
    inputImg (np.ndarray): Given input image

    Return:
    np.ndarray: Dilated image
    '''

    # # Constrói a estrutura
    # structuring_kernel = np.ones((3, 3))
    
    # # Faz um pad na imagem
    # orig_shape = img.shape
    # image_pad = np.pad(array=img, pad_width=1, mode='constant')
    # h_reduce, w_reduce = (image_pad.shape[0] - orig_shape[0]), (image_pad.shape[1] - orig_shape[1])
    
    # # Calcula as submatrizes que conterão a estrutura 3x3
    # flat_submatrices = np.array([
    #     image_pad[i:(i + 3), j:(j + 3)]
    #     for i in range(image_pad.shape[0] - h_reduce) for j in range(image_pad.shape[1] - w_reduce)
    # ])
    
    # # Substitui valores com 1 ou 0, dependendo se a estrutura "entra" na sub matriz
    # image_dilate = np.array([1 if (i == structuring_kernel).any() else 0 for i in flat_submatrices])

    # # Reshape para as dimensões originais
    # image_dilate = image_dilate.reshape(orig_shape)

    # return image_dilate

    outImg: np.ndarray = np.zeros(img.shape)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            outImg[i, j] = np.max(img[i-1:i+2, j-1:j+2])

    return outImg

def ImorphFunc(
        binaryImg: np.ndarray,
        parameterF: int) -> np.ndarray:
    '''
    Apply the morphological operation to the given image,
    according to the parameter F given (alters the order to opening to closing)

    Parameters:
    inputImg (np.ndarray): Given input image
    parameterF (int): Decides the order applied

    Return:
    np.ndarray: Image with morphology applied
    '''

    morphImg: np.ndarray = np.copy(binaryImg)

    if(parameterF == 1):
        morphImg = erosion(morphImg)
        morphImg = dilation(morphImg)
    elif(parameterF == 2):
        morphImg = dilation(morphImg)
        morphImg = erosion(morphImg)
    else:
        print("Algo de errado não está certo...")
        return None

    return morphImg

def createCoMat(
        inputImg: np.ndarray,
        pixelRef: tuple) -> np.ndarray:
    '''
    Function that calculates the co-ocurrency matrix of a given normalized mask.

    Parameters:
    inputImg (np.ndarray): Given mask
    pixelRef (tuple): Direction to follow

    Return: np.ndarray : The co-ocurrency matrix
    '''

    maxSize: int = np.max(inputImg)+1
    coMat: np.ndarray = np.zeros((maxSize, maxSize)).astype(np.uint32)

    for i in range(1, inputImg.shape[0]-1):
        for j in range(1, inputImg.shape[1]-1):
            coMat[inputImg[i, j], inputImg[i + int(pixelRef[0]), j+int(pixelRef[1])]] += 1

    return coMat

def probTransform(
    coMat: np.ndarray) -> np.ndarray:
    '''
    Auxiliary function that transforms the matrix to a probability matrix
    
    Parameters:
    coMat (np.ndarray): co-ocurrency matrix
    
    Return:
    np.ndarray: Probability matrix'''

    maxVal: int = np.sum(coMat)
    outputMat: np.ndarray = coMat/maxVal

    return outputMat

def HaralickDescriptor(
        coMat: np.ndarray) -> np.ndarray:
    '''
    Function that calculates the Haralick Descriptors to a given co-ocurrency matrix.

    Parameters:
    coMat (np.ndarray): Co-ocurrency matrix

    Return: np.ndarray : An array with the descriptors
    '''

    outputArray: np.ndarray = np.zeros(8).astype(np.float32)
    M, N = coMat.shape
    i, j = np.ogrid[0:M, 0:N]

    # Auto correlation
    outputArray[0] = (coMat*(i*j)).sum()

    # Constrast
    outputArray[1] = (coMat*(i-j)*(i-j)).sum()

    # Dissimilarity
    outputArray[2] = (coMat*(np.abs(i-j))).sum()

    # Energy
    outputArray[3] = (coMat*coMat).sum()

    # Entropy
    outputArray[4] = -(coMat[coMat > 0] * np.log(coMat[coMat > 0])).sum()

    # Homogeneity
    outputArray[5] = (coMat/(1+(i-j)*(i-j))).sum()

    # Inverse Difference
    outputArray[6] = (coMat/(1+np.abs(i-j))).sum()

    # Maximum probability
    outputArray[7] = np.max(coMat)

    return outputArray


def euclideanDistance(
        vec1: np.ndarray,
        vec2: np.ndarray) -> np.float32:
    '''
    Function that calculates the Euclidean Distance of 2 given arrays.

    Parameters:
    vec1 (np.ndarray): First array
    vec2 (np.ndarray): Second array

    Return: np.float32 : the Euclidean distance
    '''
    z = np.sum((vec1 - vec2) * (vec1 - vec2))

    return np.sqrt(z)

##################################################################################################
# Main


def main():
    '''
    Main function. Given the parameters, it will:
    Generate a grayscale image
    Binarize the image using the threshold
    Apply the morphology function
    Obtain 2 segmented masks

    Extract the features with haralick and co ocurrence
    Perform queries to rank the similarity
    '''

    # Read the inputs
    index: int = int(input())
    pixelRef: tuple = tuple(input().split(' '))
    parameterF: int = int(input())
    threshold: int = int(input())
    dataSetSize: int = int(input())
    dataSetNames: list = []

    masksDescriptors: np.ndarray = np.ndarray((dataSetSize, 16,))
        
    # Iterate for each input image
    for i in range(dataSetSize):
        imgName: str = input().rstrip()
        dataSetNames.append(imgName)

        # Open the images and apply the asked operations
        #0.6 seg
        inputImg: np.ndarray = imageio.imread(imgName).astype(np.uint8)
        grayInput: np.ndarray = RGBtoGrayScale(inputImg)
        binaryImg: np.ndarray = binarizeImg(grayInput, threshold)
        # start_time = time.time()
        ImorphImg: np.ndarray = ImorphFunc(binaryImg, parameterF)
        # end_time = time.time()
        # time_elapsed = (end_time - start_time)
        # print(time_elapsed)

        # Obtain the masks
        #0.001
        mask1: np.ndarray = np.zeros((ImorphImg.shape), np.uint8)
        mask2: np.ndarray = np.zeros((ImorphImg.shape), np.uint8)
        mask1[np.where(ImorphImg == 0)] = grayInput[np.where(ImorphImg == 0)]
        mask2[np.where(ImorphImg == 1)] = grayInput[np.where(ImorphImg == 1)]

        # Calculates the co-ocurrency matrix of mask1 and mask2
        #0.27 seg
        coMat1: np.ndarray = createCoMat(mask1, pixelRef)
        coMat2: np.ndarray = createCoMat(mask2, pixelRef)
        coMat1 = probTransform(coMat1)
        coMat2 = probTransform(coMat2)

        # Obtain Haralick descriptors and stores each in a list
        #0.02 seg
        mask1Descriptor: np.ndarray = HaralickDescriptor(coMat1)
        mask2Descriptor: np.ndarray = HaralickDescriptor(coMat2)
        masksDescriptors[i] = np.concatenate( (mask1Descriptor, mask2Descriptor), axis=None)

    # Sort the lists
    #0.0 seg
    euclDist: list = []
    for i in range(dataSetSize):
        euclDist.append(euclideanDistance(
            masksDescriptors[index], masksDescriptors[i]))
    euclDist = 1-normalize(np.array(euclDist))
    output = [x for _, x in sorted(zip(euclDist, dataSetNames))]
    # euclDist.sort(reverse=True)

    # Print
    i = 0
    print("Query: "+dataSetNames[index])
    print("Ranking:")
    for x in reversed(output):
        print("(%d)" % i, x)
        i += 1

    # fig = plt.figure(figsize=(10,10))
    # ax1 = fig.add_subplot(221)
    # ax1.imshow(binaryImg, cmap="gray")
    # ax1 = fig.add_subplot(222)
    # ax1.imshow(ImorphImg, cmap="gray")
    # ax2 = fig.add_subplot(223)
    # ax2.imshow(mask1, cmap="gray")
    # ax2 = fig.add_subplot(224)
    # ax2.imshow(mask2, cmap="gray")
    # plt.show()


if __name__ == '__main__':
    main()
