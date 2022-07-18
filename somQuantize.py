####################################################################################################
#                                                                                      #
#                                                                            #
#                                                                                                  #
#   1. adding normalized image                                                                     #
#   2. save output image                                                                           #
#   3. new plot                                                                                    #
#   4. find MSE                                                                                    #
#                                                                                                  #
#                                                                                                  #
#   How to use                                                                                     #
#   1. change img_name                                                                             #
#   2. just run !!!                                                                                #
#                                                                                                  #
#   * input image need to be square.                                                               #
#   * according to some image, codebook value is sometimes out of range [0..1].                    #
#     so I searched for the solution and found simple image normalization to prevent error.        #
#   * training and decoding are in the same iteration to minimize lines of code.                   #
#                                                                                                  #
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#import image
img_name = 'Lenna.png'
img = mpimg.imread(img_name)

################# !!!! image need to be normalized to prevent [R,G,B] value exceed 1 !!!! ##################
normalized_img = np.array((img - np.min(img)) / (np.max(img) - np.min(img)))
#print(normalized_img.shape)
# plt.imshow(normalized_img)
# plt.show()

#initialize random codebook
codebook = np.random.rand(8,8,3)
codebook_size = codebook.shape
print("codebook_size: ",codebook_size)

#declare important variables
image_size = normalized_img.shape
print("image_size: ",image_size)
alpha = 1.0
alpha_decrement_value = 1.0/256
change_alpha_at = image_size[0] * image_size[1]/256
pixel_count = 0
image_out = np.zeros((image_size[0],image_size[1],3))
decode = False

#loop through image to pick pixel and train
for i in range(0,2):
    if decode == False:
        print("Training...")
    else:
        print("Decoding...")
    #print(decode)
    for r in range(0, image_size[0]):
        for c in range(0, image_size[1]):
            if decode == True:
                pixel = normalized_img[np.newaxis,r,c]
                #print(pixel)
            else: 
                #randomly pick a pixel from image
                pixel_pos = np.random.randint(image_size[0], size=(2,1))
                pixel = normalized_img[pixel_pos[0], pixel_pos[1]]
            
            #find Best Matching Unit
            #loop through codebook to find euclidean distance
            ed_min = 100000000
            ed_min_index = [0,0]
            for r_cb in range(codebook_size[0]):
                for c_cb in range(codebook_size[1]):
                #calculate euclidean distance
                    ed_r = (pixel[0, 0] - codebook[r_cb, c_cb, 0]) ** 2
                    ed_g = (pixel[0, 1] - codebook[r_cb, c_cb, 1]) ** 2
                    ed_b = (pixel[0, 2] - codebook[r_cb, c_cb, 2]) ** 2
                    ed = (ed_r + ed_g + ed_b) ** (0.5)
                    if(ed < ed_min):
                        ed_min = ed
                        ed_min_index = [r_cb, c_cb]
            
            if decode == True:
                #decode part
                image_out[r, c, 0] = codebook[ed_min_index[0], ed_min_index[1], 0]
                image_out[r, c, 1] = codebook[ed_min_index[0], ed_min_index[1], 1]
                image_out[r, c, 2] = codebook[ed_min_index[0], ed_min_index[1], 2]
            else:
                #adjust weight of BMU
                codebook[ed_min_index[0], ed_min_index[1], 0] += alpha * (pixel[0, 0] - codebook[ed_min_index[0], ed_min_index[1], 0])
                codebook[ed_min_index[0], ed_min_index[1], 1] += alpha * (pixel[0, 1] - codebook[ed_min_index[0], ed_min_index[1], 1])
                codebook[ed_min_index[0], ed_min_index[1], 2] += alpha * (pixel[0, 2] - codebook[ed_min_index[0], ed_min_index[1], 2])

                #change learning rate
                pixel_count += 1
                if pixel_count > change_alpha_at:
                    pixel_count = 0
                    alpha -= alpha_decrement_value
                   
        #print(r)
    decode = True 
plt.imshow(codebook)
plt.show()

# save output image
mpimg.imsave("quantized_" + img_name, image_out)
# plt.imshow(image_out)
# plt.show()

###################### Mean Square Error part ########################
original = cv2.imread(img_name)
normalized_original = np.array((original - np.min(original)) / (np.max(original) - np.min(original)))
quantized = cv2.imread("quantized_" + img_name)

# split and re-merge because openCV format B,G,R 
b,g,r = cv2.split(normalized_original)       
img1 = cv2.merge([r,g,b])
b,g,r = cv2.split(quantized)
img2 = cv2.merge([r,g,b])

# calculate MSE
mse = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
mse /= float(img1.shape[0] * img1.shape[1])

###################### Visualization part ############################
# setup the figure
fig = plt.figure("MSE")
plt.suptitle("MSE: %.2f" % (mse))

# show first image
ax = fig.add_subplot(2, 2, 1)
plt.title('Original')
plt.imshow(img1)
plt.axis("off")

# show second image
ax = fig.add_subplot(2, 2, 2)
plt.title('Quantized')
plt.imshow(img2)
plt.axis("off")

# show codebook
ax = fig.add_subplot(2, 2, 3)
plt.title('Codebook')
plt.imshow(codebook)

# show the images
plt.show()