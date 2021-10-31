##  Project Proposal: Digit Recognizer
#### Group 3: Sahil Bambulkar, Rami Elzibawi and Jacob White

### What is the problem that you will be investigating? Why is it interesting?

For our CS 301 Data Science project we will be creating a digit recognizer that is able to assign labels to handwritten digits. This problem is interesting as it will allow written digits to be digitized as values within a computer. Some applications of this technology could be for professors to quickly collect grades for students after hand grading papers. Another application would be to collect dates, prices and times from historical English written documents and records.

### What data will you use? If you are collecting new data, how will you do it?

MNIST ("Modified National Institute of Standards and Technology") is an important dataset in computer vision that has laid the foundation for classification algorithms through its collection of handwritten of images. The MNIST dataset consists of 60,000 training images of handwritten digits. If necessary, we will also test from the CENPARMI, CEDAR databases handwritten digit datasets. 

### What reading will you examine to provide context and background?

We will be using Liu et al.'s paper "Handwritten digit recognition: benchmarking of state-of-the-art techniques" which has been cited 682 times. We will also use Ghosh and Maghari's paper "A Comparative Study on Handwriting Digit Recognition Using Neural Networks". These two papers build the core of our project as we will be comparing different neural network approaches. 

### What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations? You don’t have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on.

Based on our preliminary research, using a neural network approach to solve this classification problem seems best. Among those, we aim to compare three famous neural network approaches of deep neural network (DNN), deep belief network (DBN) and convolutional neural network (CNN). In this way, we can improve the implementations by picking the best of the three for our proposed solution. 

### How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

We will evaluate each approach's results based on accuracy of correctly labeling previously unseen images as well as time of execution. We can compare results by plotting each in respect to the time to complete and accuracy. This will be done in a line chart and bar chart. Our expectation is one of the neural network approaches will perform better than the others. 

## Works Cited
Liu, C. L., Nakashima, K., Sako, H., & Fujisawa, H. (2003). Handwritten digit recognition: benchmarking of state-of-the-art techniques.  _Pattern recognition_,  _36_(10), 2271-2285.

M. M. Abu Ghosh and A. Y. Maghari, "A Comparative Study on Handwriting Digit Recognition Using Neural Networks," 2017 International Conference on Promising Electronic Technologies (ICPET), 2017, pp. 77-81, doi: 10.1109/ICPET.2017.20.
