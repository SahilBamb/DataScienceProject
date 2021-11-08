##  Project Proposal: Multi-Digit Recognizer for Street Numbers
#### Group 3: Sahil Bambulkar, Rami Elzibawi and Jacob White

### What is the problem that you will be investigating? Why is it interesting?

We will be advancing the problem of digit recognizing by taking on a more difficult problem as well as improving on the current approaches effectiveness. The problem will be recognizing multi-digit numbers from Street View images. This is more challenging as the nascent version of this problem only considers digits ranged from 0-9. Adding multiple digits adds additional complexity like localization (localizing the individual digit with a bounding box)  and segmentation (segmentation is the task of separating objects of interest from each other). This is where an updated efficient approach will come in, as we will be unifying the three steps of localization, segmentation and recognition using a deep convolutional neural net on the pixels themselves. 

### What data will you use? If you are collecting new data, how will you do it?

One dataset we will be using is the **MNIST ("Modified National Institute of Standards and Technology")** which is an important dataset in computer vision that has laid the foundation for classification algorithms and we can use it for its digit images (0-9). The MNIST dataset consists of 60,000 training images of digits.

Another dataset we will be using is the **SBHN (Street View House Number)** benchmark dataset that contains 600,000 images of printed digits from the real world. This consists of house numbers and store signs. Many of the images center around the relevant digit, but may include others as well. The set consists of three sets, training, testing and an extra cleaner set. 

Though this should be enough, if necessary, we will also use the CENPARMI, CEDAR databases which also include digits.

### What reading will you examine to provide context and background?

Primarily we will be using Goodfellow et al.'s paper **Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks**. Based on Google Software Engineer Tianyu Tristan's Github repository we have found many other resources such as Mnih et al.'s paper on Recurrent Models of Visual Attention and Ba et al.'s paper on Multiple Object Recognition with Visual Attention. Our full list of gathered sources is included in our Works Cited below. 

### What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations? You don’t have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on.

We will be approaching this model based on Goodfellow et al.'s paper by implementing the DistBelief Neural Network due to its ability to manage large scale neural networks. This will consistent of unifying the localization, segmentation and recognition by using a pixel approach. Additionally, it will use a new output layer to provide conditional p

Based on our research, we believe the model will perform best with a deep architecture (with many layers) and hopefully approach human levels of digit recognizing performance. 

### How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

In the experiment, we will evaluate our approach based on the test dataset's of publicly available Street View House Numbers SVHN. But in terms of effective improvement, we will compare our efficiency to previous incarnations of multi-digit recognizers that approach 90%. Achieving human accuracy of 95% or higher will be the goal of our project. 


## Works Cited
Liu, C. L., Nakashima, K., Sako, H., & Fujisawa, H. (2003). Handwritten digit recognition: benchmarking of state-of-the-art techniques.  _Pattern recognition_,  _36_(10), 2271-2285.

M. M. Abu Ghosh and A. Y. Maghari, "A Comparative Study on Handwriting Digit Recognition Using Neural Networks," 2017 International Conference on Promising Electronic Technologies (ICPET), 2017, pp. 77-81, doi: 10.1109/ICPET.2017.20.

## Research from https://github.com/tianyu-tristan/Visual-Attention-Model

[1] Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks. Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet (2013). https://arxiv.org/abs/1312.6082

[2] Recurrent Models of Visual Attention, Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, https://arxiv.org/abs/1406.6247

[3] Multiple Object Recognition with Visual Attention, Jimmy Ba, Volodymyr Mnih, Koray Kavukcuoglu, https://arxiv.org/abs/1412.7755

[4] Reading Digits in Natural Images with Unsupervised Feature Learning , Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng on NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf

[5] Spatial Transformer Networks, Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu, https://arxiv.org/abs/1506.02025

[6] Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, Williams et al, http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

