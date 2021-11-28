

Final Presentation (video recording) 
Link to Slides: https://docs.google.com/presentation/d/1d5LgKyJn92J_xLIGte4ZKDs7vA3gpfvPEhMyAJtW2Ik/edit?usp=sharing Multi-Digit Recognizer on The Street View House Numbers (SVHN) Dataset Group 3: Sahil Bambulkar, Rami Elzibawi and Jacob White

**

# Multi-Digit Recognizer on The Street View House Numbers (SVHN) Dataset Group 3: 

### Sahil Bambulkar, Rami Elzibawi and Jacob White

## Abstract

With the prevalence of image and video data on the internet, image classification and detection is a very relevant problem. A subset of this problem involves character recognition, and while this has been largely solved for documents, this is a very limited domain and so the larger problem of detecting numbers still remains to be solved. This project will aim to build off of Goodfellow et al.’s paper to use a Convolutional Neural Network (CNN) model to predict a sequence of multi-digit numbers from street signs and house numbers. The data comes from the The Street View House Number (SVHN) Dataset - Format 1 which contains 150,000 labeled sample images. This data was used to train the CNN for numbers up to 4 digits. The reported train and test sequence accuracies are XX% for train and XX for test. 

## Limitations

Due to changes in the tensorflow library almost every code implementation of the Goodfellow et al.'s paper had been outdated and needed small to subsantial restructuring to run. Another issue was acquring the hardware neccessary to train the model, as Google Collab would take up to twenty hours and often run over its memory allocation and crash. These two issues compounded, as many times we would only find out a certain function call was outdated only after a long training period. Due to these limitations, we did not create this Preprocessing and Convlutional Neural Network (CNN) implementation from scratch, rather we will be describing a previously made implementation made by Georgia Institute of Technology scholar Binu Enchakalody. We made small changes to this code such as adjusting hyperparameters (epochs and batch size), trained a model and used it to test unseen data. 

## Introduction

The problem we are working on is correctly classifying multi-digit numbers in street level photographs coming from house numbers or Street View House Numbers (SVHN). In this case, a multi-digit number is a sequence of consecutive integer digits from 0-9. When determining accuracy, we will only be concerned with the recognizer’s ability to accurately predict every digit as well as return the correct number of digits. We will not be concerned with partial accuracy. One final note is that house numbers rarely go above lengths of 5 digits. The dataset we will be working on reflects this with very few training images with more than 5 or more digits. So the sequences can be considered of bounded length and the model should not return output in those cases. 

Similar problems have been solved, such as optical character recognition (OCR) on constrained domains like document processing as they have many working solutions that deliver predictions with human comparable accuracy. On the other hand, Arbitrary multi-character text recognition in disparate photographs has still not been mastered. The issues arise because of the variability of fonts, colors, styles, orientations and character arrangements. As well environmental factors like lighting, shadows, and resolution, motion and focus blur. 

**Why it is Important**

Multi-digit number recognition in street view images is extremely useful in our current world of digital cartography with the widespread use of online mapping applications like Google Maps and Apple Maps. With global maps from every nation on earth, the ability to set an exact address from an images pixels, combined with the geographic location of each photograph, could be extremely valuable in pinpointing the addresses of collected images. 

Another application is in [CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart)](https://support.google.com/a/answer/1217728?hl=en). CAPTCHAs are challenges whose aim is to limit access to non-human users by administering tests easily passable by humans but difficult for computers. One common CAPTCHA test is a series of malformed or slanted numbers and a text box asking users to correctly identify the number. If the multi-digit recognizer is able ability to correctly identify these numbers and bypass this, at this point prevalent, internet security measure it is of concern and should motivate the updating of these tests. 

### Overview of Results



## Related Work

### Goodfellow et al. 

The primary work of our project was Goodfellow et al.'s paper on [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf)

This paper presented a solution to solving the problem of multi-digit number recognition from street view images. Given X input image, the goal is to learn a model of P(S | X) by maximizing log P(S | X) on the training set. S, is a collection of N random variables from S1 ... SN which represent the digits of the sequence while L represents the length of the sequence. Each of these variables is discrete with a small number of possible values. L has values from 0 to 5 or above 5, and each digit has 10 possible values. This lends itself toward a softmax classifer that recieves input features extracted from X by a convolutional neural network. The softmax funciton is a function that takes in a vector of K real values and returns a vector of K values that represent probabilities: they sum to 1 and are each between 0 to 1.

 was modeled as a collection of N random variables for the elements as well as N representing the length of the sequence. Eachof the 

## Data

![img](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.3bUmFWX_Z0ext4TbBUZdvAHaBW%26pid%3DApi&f=1)

The most elementary version of digit recognizing problem is classifying the handwritten digits in the Modified National Institute of Standards and Technology (MINHST) dataset. The [MINHST handwritten dataset](http://yann.lecun.com/exdb/mnist/) consists of 60,000 black and white, centered handwritten digits ranging from 0-9 within 28 pixel by 28 pixel images. This is considered a relatively simple and straightforward image classification problem because the handwritten numbers are black and white, centered, straight and not cluttered with multiple digits. 

![img](http://ufldl.stanford.edu/housenumbers/32x32eg.png)

A more advanced version of this problem was introduced with the **[The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers) - format 2**. Images from the SVHN dataset are street sign pictures taken from Google Street View images. The SVHN dataset consists of 73257 digits for training and 26032 digits for testing as well as 531131 for extra training. **SVHN - Format 2 Dataset** consists of 32 by 32 pixel fully cropped individual (MNIST) digits. This dataset contains cropped, centered and cleaned and labeled images from street signs with digits ranging from 0 - 9. This problem is very similar to the previous dataset, but is slightly more challenging due to the addition of colors and fonts. Images also contain distracting digits that add some issues as well. 

![img](http://ufldl.stanford.edu/housenumbers/examples_new.png)

Finally we come to the dataset that was used for this problem. An undertanding of the previous comparatively easier datasets, provides good background on why this dataset was chosen for this problem. The **SVHN - Format 1 Dataset** contains multi-digit labeled street images with numbers with 0 to 5 length whose individual digits ranged in value from 0 to 9. As the two preceding examples demonstrated, this is a substantially harder problem for a number of reasons. To begin with, the multiple digits per number within the image adds exponentially more possible classes as previously there were only 0 - 9 with a single digit context. Additionally, the images are not centered or rotated, have blurring and lighting differences. 

### Preprocessing

For image normalization, mean subtraction was done for all traning and test data. 

Mean subtraction is done for all training and test samples [1][2]. 

This is followed by a feature normalization [4] using the mean and standard deviation (sd). The mean for each feature is subtracted from the training set. This is then divided by the sd. 

The same mean and sd are used on all the test images. A train, test and validation set were used here. 

Approximately 150,000 images were used for the training set which includes 30,000 true negatives. The test set contains ~13,000 samples and a validation set is randomly chosen from 10% of the training set. 

Typical mini-batch sizes are 16, 32, 64, 128, etc. The mini-batch size used here is 64(32 and 128 were tried). Each pass allows the model to take one step in the gradient descent or one update. 

The descent may appear noisy but in runs much faster especially when using a big data set. Each batch was shuffled to ensure the samples were split randomly. 

For the optimizer, Adaptive Momentum Estimation (Adam) is used in all 3 models. Adam is a very effective optimizer which is a combination of both momentum (exponential weighted average) and Root Mean Square (RMS) prop. 

Momentum can reduce the oscillations before convergence by accounting for the gradients from the previous steps. The hyper parameters for learning rate (α), momentum (β1) and RMS(β2) are recommended to be used at their default values of 0.001, 0.9 and 0.999. A decreasing learning rate when value loss plateaued was experimented with using SGD optimizer. Adam adapted better without any α tuning for the same. The weights for each layer were randomly initialized [3][4] by Keras to ensure the weights are not uniform. I experimented with using l1 and l2 norm regularizers to avoid the effects of over-fitting in the first few models. This was soon replaced by Drop outs after every other layer. This method randomly drops hidden units and measures their performance. Dropouts make the neurons less sensitive to the activation of specific neurons because that other neuron may shut down any time [3] Early stopping is used if the model loss plateaus for more than 5 epochs. This was experimented with validation loss as well. Each layer is batch normalization which like normalizing the input, helps the convergence [3][4]. This is applied before each max. pooling layer. 

## Methods

### Background of CNNs

To solve our problem of the SVHN dataset we used a CNN (convolutional neural network). There are many reasons we used CNN for our problem, firstly CNN’s take advantage of inputs being images. Images are seen as a 3d structure with those dimensions being the weight, height , and depth, depth being the RGB color channels. An advantage of having our input space being 3 dimensional is that we do not have to flatten our image pixels resulting in more information. 

Secondly CNN’s use shared weights and neurons property resulting in a smaller number of weights. Lastly CNN’s have been proven to give the best results in terms of accuracy when it comes to image recognition problems which was perfect in our scenario because the SVHN problem is an image recognition problem. 

Now that we know what the best model to use is, we should explain it. CNN's input is the raw pixel values of the images. Like we mentioned before, the third dimension in CNN is the depth which were our colors , that means that grey scale images which do not have any color are 2d because they will have no depth and colored images will be 3D. 

Now that we know about the input it takes we can talk about the layers. CNN’s have four types of layers. Convolutional layers are the building blocks of CNN’s. A convolution is an application of a filter to an input that results in an activation, repeating the application of the same filter will result in a feature map which summarizes the presence of detected features in that input. 

The next layer is the pooling layer, the function of this layer is to reduce spatial size inturn reducing the amount of parameters and computation on the machine or network this layer operates on each feature map independently, It is used for dimension reduction.

Our third layer is our Relu layer (the rectified linear activation function). This is a piecewise linear function that will output the input directly if it is positive and if it is not it will instead output 0, the purpose of using this layer is to increase the non linearity in our images. 

The final layer for CNN’s is the fully connected layer, this layer is simply feed forward neural networks. The input to these networks is the output from the final pooling or convolutional layer which is then flattened and fed into the fully connected layer, after passing through the fully connected layers , the final layer uses softmax activation function which we use to get the probabilities of the input belonging to a certain class which is the goal of our CNN. 

### Our CNN Model implementation

![image-20211127205637408](/Users/sahilbambulkar/Library/Application Support/typora-user-images/image-20211127205637408.png)

This is our project's CNN model (originally designed by Georgia Institute of Technology scholar Binu Enchakalody):

* Activation for all layers except the final dense layer was ReLu 
* We have six softmax outputs all connected to the final layer , the first represents the number of digits in the picture, for example the number 369 would have a number of digits of 3. 
* Our next four outputs would be the digits themselves since our max is 4 for the amount of digits we can identify, and finally our final output is a binary digit or not output for catching true negatives. 
* When compiling our model we used sparse categorical cross entropy for our loss.

## Experiments



## Conclusion

**Abstract: Briefly describe your problem, approach, and key results. Should be no more than 300 words.****Introduction (10%): Describe the problem you are working on, why it’s important, and an overview of your results.** **Problem** 



## Other Resources

### These other resources were useful while completing this project 

Udemy Course Deep Learning Convolutional Neural Networks in Python by Matthew Gregg 

MIT 6.S191: Convolutional Neural Networks Lecture

Stanford Lecture Series on Convolutional Neural Networks for Visual Recognition



