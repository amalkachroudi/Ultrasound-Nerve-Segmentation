
## Ultrasound-Nerve-Segmentation

Accurately identifying nerve structures in ultrasound images is a critical step in effectively inserting a patient’s pain management catheter. In this task, I am challenged to build a model that can identify nerve structures in a dataset of ultrasound images of the neck. Doing so would improve catheter placement and contribute to a more pain free future. Image segmentation aims to separate an image into anatomically meaningful regions. The objective evaluation of image segmentation methods is crucially important in order to get automated image segmentation methods accepted in clinical practice.

![myimage-alt-tag](https://raghakot.github.io/images/ultrasound/example.jpg)

# Building the Model

I used deep learning and more specifically an architecture based on CNN because they are suitable for processing images.
The network that I have used is the UNet which is composed of an encoder part in which the image is downsampled and a decoder part in which the image is upsampled in order to get eventually the predicted image that has the same size as the real mask.

![myimage-alt-tag](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)


