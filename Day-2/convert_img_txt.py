from PIL import Image
import numpy as np

# Load the image
image_path = 'image.jpg'  # Provide the path to your image
img = Image.open(image_path)

# Convert the image to RGB (in case it's not already in RGB format)
img_rgb = img.convert('RGB')

# Convert the image into a numpy array of shape (height, width, 3)
img_array = np.array(img_rgb)

# Flatten the array into a 1D array, where each pixel has 3 values: [R, G, B]
flattened = img_array.reshape(-1)

# Open a text file to store the RGB values
with open('image.txt', 'w') as f:
    # Write each pixel's RGB values space-separated
    f.write(' '.join(map(str, flattened)))

print("Image RGB data has been written to output.txt")


## Convert txt back to image in gray scale

# %matplotlib inline 
# gray_image = np.array(gray.split(" "), dtype=float)

# from matplotlib import pyplot as plt
# plt.imshow(gray_image.reshape(210, 236), interpolation='nearest',cmap='gray')
# plt.show()