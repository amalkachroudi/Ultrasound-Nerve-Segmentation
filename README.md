
# Ultrasound-Nerve-Segmentation

Accurately identifying nerve structures in ultrasound images is a critical step in effectively inserting a patient’s pain management catheter. In this task, I am challenged to build a model that can identify nerve structures in a dataset of ultrasound images of the neck. Doing so would improve catheter placement and contribute to a more pain free future. Image segmentation aims to separate an image into anatomically meaningful regions. The objective evaluation of image segmentation methods is crucially important in order to get automated image segmentation methods accepted in clinical practice.

![myimage-alt-tag](https://raghakot.github.io/images/ultrasound/example.jpg)

## Building the Model

I used deep learning and more specifically an architecture based on CNN because they are suitable for processing images.
The network that I have used is the UNet which is composed of an encoder part in which the image is downsampled and a decoder part in which the image is upsampled in order to get eventually the predicted image that has the same size as the real mask.

![myimage-alt-tag](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

### Why the U-Net ?
* The U-Net combines the location information from the downsampling path with the contextual information in 
the upsampling path to finally obtain a general information combining localisation and context, which is necessary to predict a good segmentation map.
* No dense layer, so images of different sizes can be used as input (since the only parameters to learn on convolution layers are the kernel, and the size of the kernel is independent from input image’ size).
* The use of massive data augmentation is important in domains like biomedical segmentation, since the number of annotated samples is usually limited.

## Results&Improvements

On our small dataset, the trained model achieved an accuracy of 0.9806 but this doesn’t mean that the model is good because this higher accuracy might due to the imbalanced classes( the model tended to classify all the pixels as black ).
Also our model reached 0.728 of Dice on the validation set. While this result proved quite successful in providing insights, there was still room for improvement.
In the future, we plan on augmenting our data by generating new images from our existing dataset and tuning hyperparameters via tools like cross validation. Additionally, once we have more labeled data, we will be able to further explore transfer learning options.
