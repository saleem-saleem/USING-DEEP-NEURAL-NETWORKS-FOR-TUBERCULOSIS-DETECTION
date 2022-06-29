#Before training, you'll first modify your images to be better suited for training a convolutional neural network. 
#For this task you'll use the Keras ImageDataGenerator function to perform data preprocessing and data augmentation.
#This class also provides support for basic data augmentation such as random horizontal flipping of images.
#We also use the generator to transform the values in each batch so that their mean is 0 and their standard deviation is 1 
#(this will faciliate model training by standardizing the input distribution).
#The generator also converts our single channel X-ray images (gray-scale) to a three-channel format by repeating the values in the image across all channels 
#(we will want this because the pre-trained model that we'll use requires three-channel inputs).

image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)


 #Build a separate generator fo valid and test sets Now we need to build a new generator for validation and testing data.

# Why can't use the same generator as for the training data?
#Look back at the generator we wrote for the training data.It normalizes each image per batch, meaning thatit uses batch statistics.
#We should not do this with the test and validation data, since in a real life scenario we don't process incoming images a batch at a time (we process one image at a time).
#Knowing the average per batch of test data would effectively give our model an advantage (The model should not have any information about the test data).
#What we need to do is to normalize incomming test data using the statistics computed from the training set.

train = image_generator.flow_from_directory(train_dir, 
                                            batch_size=8, 
                                            shuffle=True, 
                                            class_mode='binary',
                                            target_size=(320, 320))

validation = image_generator.flow_from_directory(val_dir, 
                                                batch_size=1, 
                                                shuffle=False, 
                                                class_mode='binary',
                                                target_size=(320, 320))

test = image_generator.flow_from_directory(test_dir, 
                                            batch_size=1, 
                                            shuffle=False, 
                                            class_mode='binary',
                                            target_size=(320, 320))

sns.set_style('white')
generated_image, label = train.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, one single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
sns.distplot(generated_image.ravel(),
             label=f"Pixel Mean {np.mean(generated_image):.4f} & Standard Deviation {np.std(generated_image):.4f}", 
             kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
