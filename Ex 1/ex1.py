# Aluno: Dalton Hiroshi Sato
# nUSP: 11275172
# SCC0251 - Processamento de Imagens
# Ano: 2022
# Semestre: 1º
# Assignment 1 - Image Generation


from matplotlib import image
import numpy as np
import imageio
import random
import matplotlib.pyplot as plt

# Normalize from 0 to 1 (float)
def normalize(scene_image: np.ndarray):
    scene_image = (scene_image - np.min(scene_image[:]))/(np.max(scene_image[:]) - np.min(scene_image[:]))
    return scene_image

# From a scene image, it downsamples to a digital image, given a scale
def sampling_scale(scene_image: np.ndarray, digital_image: np.ndarray, scale: int):
    digital_image[0, 0] = scene_image[0, 0]

    for x, y in np.ndindex(digital_image.shape):
        digital_image[x, y] = scene_image[x*scale, y*scale]

    return digital_image

# Main function to print the aproximate error to the example given
def RSE (digital_image: np.ndarray, file_name: str):
    file_name = file_name.rstrip()
    image_comparison = np.load(file_name)
    
    sum: int = 0
    for x, y in np.ndindex(digital_image.shape):
        sum += ( int((digital_image[x, y]) - int(image_comparison[x, y]))**2 )

    return np.sqrt((sum))

# Secondary function to print RSE error
def print_RSE (digital_image: np.ndarray, reference_file_name: str):
    print("%.4f" % RSE(digital_image, reference_file_name))


# Shift bit wise, given number of bits to shift (maximum of 7 bits - color spectre)
def shift_bitwise(digital_image: np.ndarray, bits_ppixel: int):
    digital_image = digital_image.astype(np.uint8)
    digital_image = digital_image >> (8-bits_ppixel)
    
    return digital_image

# Change max values of "pixels" in the image
def change_max_value(scene_image: np.ndarray, max_value: int):
    scene_image = scene_image * max_value

    return scene_image

# Main function to create the digital image
def digital_image_generation(scene_image: np.ndarray, scene_size: int, digital_image: np.ndarray, scale: int, bits_ppixel: int):
    digital_image = sampling_scale(scene_image, digital_image, scale)
    digital_image = normalize(digital_image)

    digital_image = change_max_value (digital_image, 255)

    digital_image = shift_bitwise(digital_image, bits_ppixel)

    return digital_image

# Auxiliary to plot the image generated
# Deactivated to run.codes purposes
def plot_image(digital_image: np.ndarray):
    plt.figure()
    plt.imshow(digital_image, cmap="gray")
    plt.show()

####################################################################################################

def function1(scene_image: np.ndarray) :

    for x, y in np.ndindex(scene_image.shape):
        scene_image[x, y] = (x*y) + (2*y)

    return scene_image
    

def function2(scene_image: np.ndarray, q_parameter: int):
    
    for x, y in np.ndindex(scene_image.shape):
        scene_image[x, y] = np.absolute(np.cos(x/q_parameter) + 2*np.sin(y/q_parameter) )

    return scene_image

def function3(scene_image: np.ndarray, q_parameter: int) :
    
    for x, y in np.ndindex(scene_image.shape):
        scene_image[x, y] = np.absolute( 3*(x/q_parameter) - ((y/q_parameter)**(1/3)) )

    return scene_image


def function4(scene_image: np.ndarray, random_seed: int) :

    random.seed(random_seed)
    for x, y in np.ndindex(scene_image.shape):
        scene_image[x, y] = random.random()

    return scene_image


def function5(scene_image: np.ndarray, scene_size: int, random_seed: int) :
    
    random.seed(random_seed)
    posx: int = 0
    posy: int = 0
    scene_image[posx, posy] = 1

    for i in range(scene_size**2):
        posx = (posx + random.randint(-1, 1))%scene_size
        posy = (posy + random.randint(-1, 1))%scene_size
        scene_image[posx, posy] = 1

    return scene_image


########################################################################

def main():
    reference_file_name: str = input()
    scene_size: int = int(input())
    function_used: int = int(input())
    q_parameter: int = int(input())
    digital_image_size: int = int(input())
    bits_ppixel: int = int(input())
    random_seed: int = int(input())

    # Create a "blank" image
    scene_image = np.zeros((scene_size, scene_size))

    if(function_used == 1):
        scene_image = function1(scene_image)
    elif(function_used == 2):
        scene_image = function2(scene_image, q_parameter)
    elif(function_used == 3):
        scene_image = function3(scene_image, q_parameter)
    elif(function_used == 4):
        scene_image = function4(scene_image, random_seed)
    elif(function_used == 5):
        scene_image = function5(scene_image, scene_size, random_seed)
    else:
        print("Parâmetro de função incorreto\n")
    

    scene_image = normalize(scene_image)
    scene_image = change_max_value (scene_image, 65535)

    # Digital image. A "photo" taken from the scene
    digital_image = np.zeros((digital_image_size, digital_image_size))

    scale = int(scene_size/digital_image_size)

    digital_image = digital_image_generation(scene_image, scene_size, digital_image, scale, bits_ppixel)

    print_RSE(digital_image, reference_file_name)

    plot_image(digital_image)

if __name__ == "__main__":
    main()