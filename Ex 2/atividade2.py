# Nome: Savio Duarte Fontes
# Nusp: 10737251
# SCC0251-101-2022 Processamento de Imagens

# Assignment 2 - Image enhancement and filtering


# Packages
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Functions of the program

# Method 1 - Limiarization
#   the image will receive a black or white value according to a threshold
#
#   the threshild will be calculated up to the difference between the actual threshold and the last is 0.5
#   after that, the image_output receives white if the pixel value is bigger than the threshold
def Method1(img):
    #input threshold
    threshold = int(input())
    
    M, N = img.shape

    #threshold calculation
    g1, g2 = GroupAverage(img,threshold)
    t = 1./2*(g1+g2)

    while(abs(t-threshold) >= 0.5):
        threshold = t

        g1, g2 = GroupAverage(img,threshold)
        t = 1./2*(g1+g2)

    #output
    img_output = np.zeros([M, N]).astype(np.uint8)
    img_output[np.where(img > t)] = 255

    return img_output

# Method 2 - Filtering 1d
#   each pixel is replaced with the result of applying the filter on the neighboring pixels
#
#   the image is vectorized
#   the vector will be wraped with its elements
#   after that, the image_output[x,y] receives the calculation of the given filter on the neighboring pixels
def Method2(img):

    #input of the filter_size and the filter
    filter_size = int(input())
    filter_w = np.zeros(filter_size,float)
    
    filter_input = np.array(input().split())
    for x in range(filter_size):
        filter_w[x] = float(filter_input[x])
    
    M, N = img.shape

    #image vectorizing
    img_reshape = np.reshape(img,M*N).astype(float)

    #ratio is the number of elements that will be padded for each side
    ratio = int(filter_size/2)
    
    # manually padding the array
    #img_aux = np.zeros((M*N+2*ratio),float)
    #img_aux[ratio:M*N+ratio] = img_reshape    
    #img_aux[0:ratio] = img_reshape[M*N-ratio:]
    #img_aux[(M*N+ratio):] = img_reshape[:ratio]


    # padding with numpy.pad()
    img_aux = np.pad(img_reshape,(ratio,),'wrap')

    #output
    img_out = np.zeros((M*N),float)
    for x in range(M*N):
        for y in range(filter_size):
            img_out[x] += img_aux[x+y]*filter_w[y]

    img_out = np.reshape(img_out,(M,N))
    
    #image normalizing
    img_out = Normalize(img_out)
    img_out = (img_out*255).astype(np.uint8)

    return img_out


# Method 3 - Filtering 2d
#   each pixel is replaced with the result of applying the filter on the neighboring pixels
#
#   the image is padded with the elements of its edges
#   after that, the image_output[x,y] receives the calculation of the given filter on the neighboring pixels
def Method3(img):

    #input of the filter_size and the filter
    filter_size = int(input())
    filter_w = np.zeros((filter_size,filter_size),float)
    
    for x in range(filter_size):
        filter_input = np.array(input().split())
        for y in range(filter_size):
            filter_w[x,y] = float(filter_input[y])

    M, N = img.shape

    #ratio is the number of elements that will be padded for each side
    ratio = int(filter_size/2)

    #image padding
    img_aux = np.pad(img,ratio,'edge')

    #output
    img_out = np.zeros((M,N),float)
    for x in range(M):
        for y in range(N):
            for k in range(filter_size):
                for l in range(filter_size):
                    img_out[x,y] += img_aux[x+k,y+l]*filter_w[k,l]

    #image normalizing
    img_out = Normalize(img_out)
    img_out = (img_out*255).astype(np.uint8)

    return Method1(img_out)

# Method 4 - Median Filter
#   each pixel is replaced with the median value of neighboring pixels
#
#   the image is padded with zeros
#   after that, the image_output[x,y] receives the median value of an array2d_cell that has the size of the filter
def Method4(img):

    #input of filter size
    filter_size = int(input())
    
    M, N = img.shape

    #ratio is the number of elements that will be padded for each side
    ratio = int(filter_size/2)

    #image padding
    img_aux = np.pad(img,ratio,'constant',constant_values=(0))

    #output
    img_out = np.zeros((M,N),np.uint8)
    for x in range(M):
        for y in range(N):
            img_out[x,y] = median(img_aux[x:x+filter_size,y:y+filter_size],filter_size)
            #function np.median gives TLE on run.codes

    
    # its not necessary normalize, because the img_out has values between 0-255
    # the median operation does not generate a new interval of values     
    
    #img_out = Normalize(img_out)
    #img_out = (img_out*255).astype(np.uint8)
    
    return img_out

# Function that returns the median value of an array2d that was reshaped to one-dimensional vector and sorted
def median(array2d,size):

    array = np.reshape(array2d,size*size)
    array_sorted = np.sort(array)
    
    if(size % 2 == 0):
        return int((array_sorted[int(size*1.5-1)]+array_sorted[int(size*1.5)])/2)
    else:
        return array_sorted[int(size*1.5)]

# Function that returns the average of two groups separated by a threshold
def GroupAverage(img, threshold):

    g1 = np.average(img[np.where(img > threshold)])
    g2 = np.average(img[np.where(img <= threshold)])

    return g1, g2

# Function that normalizes an img from 0 to 1
def Normalize(img):
    img = (img - np.min(img[:]))/(np.max(img[:]) - np.min(img[:]))

    return img

# Function that gives the RMSE between the image_input and image_output
def RMSE(imgi, imgo):
    M, N = imgi.shape

    imgi = imgi.astype(float)
    imgo = imgo.astype(float)

    rmse_aux = 0
    for x in range(M):
        for y in range(N):
            rmse_aux += (imgi[x,y]-imgo[x,y])**2
    
    rmse_aux = rmse_aux / (M*N)
    
    return np.sqrt(rmse_aux)

# Function that creates a switch to choose the method of the problem
def SwitchMethod(method, img):
    if(method == 1):
        return Method1(img)
    if(method == 2):
        return Method2(img)
    if(method == 3):
        return Method3(img)
    if(method == 4):
        return Method4(img)

#######################################################################

def main():
    
    # Input
    filename = str(input()).rstrip()
    method = int(input())

    #opening of the image_input
    image_input = imageio.imread(filename)

    #generating the output of the methods
    image_output = SwitchMethod(method, image_input)

    # Ploting the images for debug
    #plt.subplot(121)
    #plt.imshow(image_input, cmap="gray")
    #plt.subplot(122)
    #plt.imshow(image_output, cmap="gray")
    #plt.show()

    # Compare image_input with image_output
    rmse = RMSE(image_input, image_output)
    format_rmse = "{:.4f}".format(rmse)
    print(format_rmse)

    return

if __name__ == "__main__":
    main()