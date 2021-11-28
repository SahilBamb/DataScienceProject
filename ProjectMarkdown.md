

Final Presentation (video recording) 
Link to Slides: https://docs.google.com/presentation/d/1d5LgKyJn92J_xLIGte4ZKDs7vA3gpfvPEhMyAJtW2Ik/edit?usp=sharing Multi-Digit Recognizer on The Street View House Numbers (SVHN) Dataset Group 3: Sahil Bambulkar, Rami Elzibawi and Jacob White

**

# Multi-Digit Recognizer on The Street View House Numbers (SVHN) Dataset Group 3: 

### Sahil Bambulkar, Rami Elzibawi and Jacob White

## Abstract

With the prevalence of image and video data on the internet, image classification and detection is a very relevant problem. A subset of this problem involves character recognition, and while this has been largely solved for documents, this is a very limited domain and so the larger problem of detecting numbers still remains to be solved. This project will aim to build off of Goodfellow et al.’s paper to use a Convolutional Neural Network (CNN) model to predict a sequence of multi-digit numbers from street signs and house numbers. The data comes from the The Street View House Number (SVHN) Dataset - Format 1 which contains 150,000 labeled sample images. This data was used to train the CNN for numbers up to 4 digits. The reported train and test sequence accuracies are XX% for train and XX for test. 

## Limitations

Due to changes in the tensorflow library almost every code implementation of the Goodfellow et al.'s paper had been outdated and needed small to subsantial restructuring to run. Another issue was acquring the hardware neccessary to train the model, the paper implementation used a distributed DisetBelief implementaton running on over 100 computers over six days and most amateur implementations ran the training over ten hours. Using Google Collab, we had our training take up to twenty hours and often run over its memory allocation and crash. These two issues compounded, as many times we would find out a certain function call was outdated only after a long training period meaning wasted productivity. Due to these limitations, we did not create this Convlutional Neural Network (CNN) implementation from scratch, rather we will be describing a slightly changed version of a previously made implementation made by Georgia Institute of Technology scholar Binu Enchakalody. We made small changes to this code such as adjusting hyperparameters (epochs and batch size), fixed code issues and used it to train a model and used it to test unseen data. 

## Introduction

The problem we are working on is correctly classifying multi-digit numbers in street level photographs coming from house numbers or Street View House Numbers (SVHN). In this case, a multi-digit number is a sequence of consecutive integer digits from 0-9. When determining accuracy, we will only be concerned with the recognizer’s ability to accurately predict every digit as well as return the correct number of digits. As house numbers rarely go above lengths of 5 digits. The dataset we will be working on reflects this with very few training images with more than 5 or more digits. So the sequences can be considered of bounded length and the model should not return output in those cases. 

Similar problems have been solved, such as optical character recognition (OCR) on constrained domains like document processing as they have many working solutions that deliver predictions with human comparable accuracy. On the other hand, Arbitrary multi-character text recognition in disparate photographs has still not been mastered. The issues arise because of the variability of fonts, colors, styles, orientations and character arrangements. As well environmental factors like lighting, shadows, and resolution, motion and focus blur. 

**Why it is Important**

Multi-digit number recognition in street view images is extremely useful in our current world of digital cartography with the widespread use of online mapping applications like Google Maps and Apple Maps. With global maps from every nation on earth, the ability to set an exact address from an images pixels, combined with the geographic location of each photograph, could be extremely valuable in pinpointing the addresses of collected images. 

Another application is in [CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart)](https://support.google.com/a/answer/1217728?hl=en). CAPTCHAs are challenges whose aim is to limit access to non-human users by administering tests easily passable by humans but difficult for computers. One common CAPTCHA test is a series of malformed or slanted numbers and a text box asking users to correctly identify the number. If the multi-digit recognizer is able ability to correctly identify these numbers and bypass this, at this point prevalent, internet security measure it is of concern and should motivate the updating of these tests. 

### Overview of Results

We were able to achieve an accuracy of XX on. Ths specifics of these results will be described more indepth below in the experiments section. 

## Related Work

### Goodfellow et al. 

The primary work of our project was Goodfellow et al.'s paper on [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) This paper presented a solution to solving the problem of multi-digit number recognition from street view images. Given X input image, the goal is to learn a model of P(S | X) by maximizing log P(S | X) on the training set. S, is a collection of N random variables from S1 ... SN which represent the digits of the sequence while L represents the length of the sequence. Each of these variables is discrete with a small number of possible values. L has values from 0 to 5 or above 5, and each digit has 10 possible values. This lends itself toward a softmax classifer that recieves input features extracted from X by a convolutional neural network. The softmax funciton is a function that takes in a vector of K real values and returns a vector of K values that represent probabilities: they sum to 1 and are each between 0 to 1. 

 was modeled as a collection of N random variables for the elements as well as N representing the length of the sequence. Eachof the 

## Data

<p align="center">  
 <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.3bUmFWX_Z0ext4TbBUZdvAHaBW%26pid%3DApi&f=1" alt="drawing"/>
</p>
The most elementary version of digit recognizing problem is classifying the handwritten digits in the Modified National Institute of Standards and Technology (MINHST) dataset. The [MINHST handwritten dataset](http://yann.lecun.com/exdb/mnist/) consists of 60,000 black and white, centered handwritten digits ranging from 0-9 within 28 pixel by 28 pixel images. This is considered a relatively simple and straightforward image classification problem because the handwritten numbers are black and white, centered, straight and not cluttered with multiple digits. 
<p align="center"> 
<img src="http://ufldl.stanford.edu/housenumbers/32x32eg.png" alt="drawing" width="300"/>
</p>
A more advanced version of this problem was introduced with the **[The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers)**. Images from the SVHN dataset are street sign pictures taken from Google Street View images. The SVHN dataset consists of 73257 digits for training and 26032 digits for testing as well as 531131 for extra training. **SVHN - Format 2 Dataset** consists of 32 by 32 pixel fully cropped individual (MNIST) digits. This dataset contains cropped, centered and cleaned and labeled images from street signs with digits ranging from 0 - 9. This problem is very similar to the previous dataset, but is slightly more challenging due to the addition of colors and fonts. Images also contain distracting digits that add some issues as well. 
<p align="center"> 
<img src="http://ufldl.stanford.edu/housenumbers/examples_new.png" alt="drawing" width="300"/>
</p>
Finally we come to the dataset that was used for this problem. An undertanding of the previous comparatively easier datasets provides good background on why this dataset was chosen for this problem. The **SVHN - Format 1 Dataset** contains multi-digit labeled street images, in color, with numbers with 0 to 5 length whose individual digits ranged in value from 0 to 9. As the two preceding examples demonstrated, this is a substantially harder problem for a number of reasons. To begin with, the multiple digits per number within the image adds exponentially more possible classes as previously there were only 0 - 9 with a single digit context. Additionally, the images are not centered, each can be rotated, have blurring and have lighting differences. Each image is also of variable size. **Approximately 150,000 images were used for the training set which includes 30,000 true negatives. The test set contains ~13,000 samples and a validation set is randomly chosen from 10% of the training set.** 

### Preprocessing

#### Image Normalization 

For the reasons outlined with the data, a degree of preprocessing was required to be applied to make these images suitable for best performance within the CNN. For image normalization, mean subtraction was done for all traning and test samples followed by feature normalizaiton using the mean and standard deviation. This occurs as the mean for each feature is subtracted from the respective set and then divided by the standard deviation. This applies, with the same mean and standard deviation, for the training, test and validation set. 

#### General Data Preperation and Premade Bounding Boxes

**The SVHN - Format 1 Dataset** provides bounding box information for each digit of its sample files. Before inputting into the preprocessing, for ease of use this data was combined into one file. This was done as the test, traning and validation data were combined with each of their respective images, bounding box information and label information into an easily accessible file rather than load and reference each of them seperately. Also of note, Bounding Box dimensions were used to crop a 30% padded region around each box. This is also applies to post-detection images. 

Additional Data Preperation includes:

* Resized to 48 x 48 pixel BGR and grayscale (although our model only used BGR)
* Binary Digit classifer was created to detect if it was a digit or not for detection purposes
* 5+ digits were ommitted for lack of data and becasue they rarely occur in real life. 

#### Detection algorithm

Premade bounding boxes are obviously not available for non-SVHN input data and thus we also include scaling, localization and sliding windows. localization is done as Image gradients and magnitude are used to create a binary mask which ignores any parts of the image without major gradient changes and avoids image elements above/below a certain size. Scaling occurs as the binary image / image is scaled ten levels by 80% each time (compared to the previous image). A 48 x 48 sliding window runs through the sample image at steps of up to 5 and ignores windows without relevant gradients. The classifier additionally removes window candidates with high likelihood of not being a digit. A final check is done for the overall confidence of it being a digit. If this is less than 70%, it is again ignored. If all of these requirements are satisifed, a local bounding box (along with liklihood) are saved. 

The preceding process of locazliaton and sliding windows often causes multiple hits so a probability mask is created using the saved bounding boxes and their predictions. One issue, espescially when working with digits, is that it will fail when close to each other. Thus, we should use non-maxima suppression to remove invalid boxes based on overlap scores. The final rsized versions of these boxes are passed to the model and if the confidence is over 80% a bounding box and digit are drawn onto the final image. 

## Methods

### CNN Model Explained

To solve our problem of the SVHN dataset we used a CNN (convolutional neural network). There are many reasons we used CNN for our problem, firstly CNN’s take advantage of inputs being images. Images are seen as a 3D structure with those dimensions being the weight, height, and depth, depth being the RGB color channels. An advantage of having our input space being 3 dimensional is that we do not have to flatten our image pixels resulting in more information. 

Secondly CNN’s use shared weights and neurons property resulting in a smaller number of weights. Lastly CNN’s have been proven to give the best results in terms of accuracy when it comes to image recognition problems which was perfect in our scenario because the SVHN problem is an image recognition problem. 

Now that we know what the best model to use is, we should explain it. CNN's input is the raw pixel values of the images. Like we mentioned before, the third dimension in CNN is the depth which were our colors , that means that grey scale images which do not have any color are 2D because they will have no depth and colored images will be 3D. 

Now that we know about the input it takes we can talk about the layers. CNN’s have four types of layers. Convolutional layers are the building blocks of CNN’s. A convolution is an application of a filter to an input that results in an activation, repeating the application of the same filter will result in a feature map which summarizes the presence of detected features in that input.

### **Convolutional layer** 

Convolutional layers are the building blocks of CNN’s. A convolution is an application of a filter to an input that results in an activation, repeating the application of the same filter will result in a feature map which summarizes the presence of detected features in that input. 

### **Pooling layer**

The function of the pooling layer is to reduce spatial size inturn reducing the amount of parameters and computation on the machine or network, it accomplishes this using the MAX function. This layer operates on each feature map independently, It is used for dimension reduction. This layer operates on depths independently so it takes the depths slice by slice. The figure below illustrates this very well

![image](https://user-images.githubusercontent.com/42818731/143778587-73ae5b4a-d54e-4a23-830a-b333147f4d83.png)


### **Activation layer** 

The activation function is a node that is put at the end of or in between Neural Networks. They help to decide if the neuron would fire or not. In most cases and in ours we use ReLu (the rectified linear activation function). This is a piecewise linear function that will output the input directly if it is positive and if it is not it will instead output 0, the purpose of using this layer is to increase the non linearity in our images. 

### **Fully Connected Layer**

The final layer for CNN’s is the fully connected layer, this layer is simply feed forward neural networks. The input to these networks is the output from the final pooling or convolutional layer which is then flattened and fed into the fully connected layer, after passing through the fully connected layers , the final layer uses softmax activation function which we use to get the probabilities of the input belonging to a certain class which is the goal of our CNN.

![image](https://user-images.githubusercontent.com/42818731/143780279-dc7e27ae-6c68-4fa5-9655-d0346f7b36e8.png)

The figure above is a screenshot of a running demo of a CONV layer taken from the [Stanford University Convolutional Networks courses page](https://cs231n.github.io/convolutional-networks/), it will help us better describe filters.

While using CNN's multiple filters are taken to slice through an image and map them one by one in order to learn different portions of the image. That's why the figure above is so important because it helps us visualize this. Its as if we are scanning our input volume reading in all the regions.

### Our layers and filters

For layer 1 in our designed model we have 2 x (16 filters 3x3), we can easily break this down by looking at the picture we have provided above. We can think of the number outside the parentheses as the amount of filters that are being used for this layer in our current layer , layer 1 we have a two in that place so it's actually exactly like in our figure above we would have filters W0, and W1 as well but if we take another layer in our model for example layer 4 instead of a 2 we have a 3 so we would instead have 3 filters W0, W1, and W2. The output volume is actually the dot product of all these filters so for layer 4 it would be the dot product of W0,W1,and W2.

### **Filters Per Layer**

* Layer’s 1,2,3,6,8 = 2 filters
* Layer’s 4,5,7 = 3 filters 

Now that we understand what the first digit represents we can now move into what's in the parentheses. If you look at the figure again you can see that we have a column for the filters and in that column we have a label W0[ : , : , 0] and it goes until we have W0[ : , : , 2] this last number the 2 represents our first number in the parentheses so in our situation for layer 1 we have 16 filters so we would run from 0 to 16, so unlike the figure above we would only have 2 rows for our filters we would instead have 16. In our model we used many different filter sizes for our layers. I believe this is our depth.

### Depth of filters

- Layer 1 = 16
- Layer 2 = 32 
- Layer 3 = 48 
- Layer 4 = 64 
- Layer 5 = 128 
- Layer’s 6,7 = 256 
- Layer 8 = 512 

We have almost completely broken down filters and conv layer operations , the last thing that we have to talk about is the filter sizes. In our figure we have a filter size of 3 by 3 meaning we have 3 rows and 3 columns, just like we have in layer 1 of our model which is also 3x3. The figure does a good job of demonstrating how the filters read in from the input volume.

### Row and Column Size

* Layer 1,2,3,4 = 3x3 

* Layers 5,6,7,8 = 5x5

### Our CNN Model implementation
<p align="center"> 
<img src="https://user-images.githubusercontent.com/42818731/143778616-cd2f4f02-57ed-4fd5-95ef-a3a1a1d97c97.png" alt="drawing"/>
</p>

This is our project's CNN model (architecture originally designed by Georgia Institute of Technology scholar Binu Enchakalody):

* Activation for all layers except the final dense layer was ReLu 
* We have six softmax outputs all connected to the final layer , the first represents the number of digits in the picture, for example the number 369 would have a number of digits of 3. 
* Our next four outputs would be the digits themselves since our max is 4 for the amount of digits we can identify, and finally our final output is a binary digit or not output for catching true negatives. 
* When compiling our model we used sparse categorical cross entropy for our loss.

* Mini batch sizes used are 64
* Adaptive Momentum Estimation (Adam) is used in all 3 models. Adam is a very effective optimizer which is a combination of both momentum (exponential weighted average) and Root Mean Square (RMS) prop. 
* The hyper parameters for learning rate (α), momentum (β1) and RMS(β2) are recommended to be used at their default values of 0.001, 0.9 and 0.999. 
* Weights for each layer were randomly generated by Keras 
* Overfitting was accounted for by Dropout Layers which randomly drops hidden units and measures their performance
* Early stopping was implemented if the loss does not effectively move for more than 5 epochs
* Each layer has batch normalization to help the converage (applied before each pooling layer)

## Experiments

We trained our model through 25 epochs and with a batch size of 64. We obtained a training loss of 1.536 and an accuracy of 97.92. We then compared our model and metrics with a pre-trained VGG model. The VGG model ended up having better metrics compared to our model by a significant margin. After we trained our model we ran a set of street view house numbers through our model and the VGG model. Both models didn’t correctly guess every number, and had problems in some images detecting digits as well as wrongly classifying digits.

The hardest part for the model was finding where to put the bounding box, which determines where the digits are located. This is challenging for the model when it is introduced to new images where it doesn’t know where the digits are located, and has to determine where to put the bounding box. The model can easily be confused with objects that resemble numbers and put the bounding boxes in areas with no numbers, also it can detect where the digits are but only put a bounding box around a few digits missing some.


### Examples from Our Model
<img src="https://user-images.githubusercontent.com/42818731/143779932-0a60ee62-6ff7-4a45-95f1-97e58721dacb.png" alt="drawing" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779933-a03e50e1-5bed-41eb-ae7e-9939a1e8f793.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779936-df766841-b88c-4f78-b430-6e2642daaf85.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779937-45b750ba-c2ee-4bd5-ac2c-28e9a632a76a.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779939-e63650fa-c75b-4007-8db5-649519685f70.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779940-64fe6fb0-892b-44f3-bdcc-dbae41db462b.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779941-b25ee199-7a26-4ef1-849d-b0fedcb10210.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779943-14a2ecfa-1407-46a2-aacb-16a0090ec6b6.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143779944-f16e8c4a-7529-44c4-9592-303c1f5b3332.png" alt="drawing" height = "200" width="200"/>

### Examples from VGG Model

<img src="https://user-images.githubusercontent.com/42818731/143780450-6204df04-8b09-4bac-abd1-2f206af59214.png" alt="drawing" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780451-9da8f3ee-c73c-4fd5-be2c-d5edf8773534.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780452-12a0419e-b24e-4b38-9449-6c3c307c1a7d.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780454-bdfb72d9-a187-4bdc-807c-07e0912f859b.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780455-912221b0-f930-4446-ac45-ceac98af904f.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780456-741429d1-f6b2-428c-b856-0d11a78882ea.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780457-31926f69-72a4-430e-810f-4ad0b0fba02d.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780458-6ca46eb4-cf99-460c-b0e1-57512133ab1d.png" alt="drawing" height = "200" width="200"/> <img src="https://user-images.githubusercontent.com/42818731/143780459-fb09a1a2-8122-4f69-bd57-43c02d40a671.png" alt="drawing" height = "200" width="200"/>

Our model ended up performing worse but still was able to classify some digits.


The VGG model also made mistakes but has overall better accuracy.

Both models struggled with the localization issue in finding where to put the bounding box, this is where most of the incorrect classifications happened. When our model was able to correctly place the bounding box it was able to classify digits fairly accurately, however it performed worse in placing the bounding box compared to the VGG model. However the VGG model also wasn’t perfect and made mistakes in placing the bounding box.

![img](https://lh6.googleusercontent.com/GlJM8tbuaJZhEiV0LUPrhGnEFFB7CDa3fon5_EkrHrNbUpQ6uwOmAILXSI7v7brO7sJG8WOBfIcoqyw23_ti8EKCV_0uPEWuY7oV4ulyHi_rrGBB4cRfpUm83--BZKoRyWyBlJwQ)

From the graphs you can see that the VGG model had better accuracy and loss, although interestingly our model’s validation loss was closer to the train loss compared to the VGG model.

| Model      | Train Loss | Test Loss | Val Loss | Train Seq. Accuracy | Val Seq. Accuracy | Test Seq. Accuracy |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| VGG Pre Train  | 0.17 | 0.76 | 0.88 | 96.62 | 87.91 | 91.24 | 
| Our Model   | 1.536 | 2.5314 | 1.567 | 87.02 | 82.16 | 80.53 | 

Again seeing the loss values you see the VGG model performing better. The accuracy numbers are a bit deceptive, as it seems our model did better than the VGG model looking at the numbers alone. Putting it in perspective their model was run on much more images and also included more images with vertical number orientations which both models have trouble classifying correctly. If we ran the VGG model on the same data as our model it would have performed better than ours, which we saw when we ran 15 sample images through both models. The biggest factor holding our model back was our limited compute power and time. The VGG model was trained on a much larger number of images, which due to our limited memory and computing power we could not replicate. The main issue for our model was placing the bounding box in new images we fed it, which due to our smaller training data set it was not as good in placing the bounding box. Although our model’s performance was not as good, taking into consideration the vast difference in computing power our model still performed well in our experiments.

## Conclusion

**Abstract: Briefly describe your problem, approach, and key results. Should be no more than 300 words.****Introduction (10%): Describe the problem you are working on, why it’s important, and an overview of your results.** **Problem** 


## Resources

##### These other learning resources were also useful while completing this project 

* [Udemy Course: Deep Learning Convolutional Neural Networks in Python by Matthew Gregg](https://www.udemy.com/course/deep-learning-convolutional-neural-networks-theano-tensorflow/) 

* [MIT 6.S191: Convolutional Neural Networks Lecture](https://www.youtube.com/watch?v=iaSUYvmCekI)

* [Stanford Lecture Series on Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/watch?v=vT1JzLTH4G4)
* [Evolution Of Object Detection Networks Course](https://www.youtube.com/playlist?list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S)



