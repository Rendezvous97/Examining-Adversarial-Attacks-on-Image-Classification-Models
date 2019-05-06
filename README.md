# Network Architecture Specific Adversarial Attacks: *Variational Auto-Encoders Vs. Convolutional Neural Networks*

## I. Introduction
Current research on adversarial examples is largely focussed on deriving general defences against these attacks for all ML models irrespective of their architecture. In contrast to this methodology, we believe that each network architecture needs to be examined separately in order to make effective and specialized defensive capabilities. We must analyze the robustness of each architecture in isolation against different types of adversarial examples to understand the extent to which they are susceptible. Therefore, In this paper, we examine the extent to which Variational Auto-Encoders (Convolutional and Vanilla) and Convolutional Neural Networks (CNNs) are vulnerable to several gradient-based attacks on two types of datasets —  high pixel density (Labelled Faces in the Wild dataset) and low pixel density (MNIST). Our aim is to review the confidence of each attack, its validity and hence, the degree of effectiveness of the attack taking place for both types of architectures. Additionally, we also examine the role siamese networks could potentially play in creating more secure and robust systems. 

## II. Background
Deep neural networks are known for their high predictive capabilities, especially in comparison to humans. Yet, Szegedy et. al. (2014) discovered that this does not hold true in all contexts. They proved that several machine learning models, including state-of-the-art deep neural networks, are susceptible to adversarial examples, i.e, “ inputs to machine learning models designed by an adversary to cause an incorrect output” (Quin, et. al.). Given hat Machine Learning models are becoming ubiquitous, there is an increasing urgency for us to evaluate their susceptibility to these attacks, especially when the risks associated are high. In the context of image classification, these adversarial examples could lead to potentially harmful situations. For example, autonomous cars could be targetted to classify a sticker as a stop sign which can cause the car to brake unexpectedly (Goodfellow, et al. 2015). Similarly, face recognition and other biometric models, used to prevent unauthorized access to sensitive databases, could be fooled into allowing access via adversarial examples. In these cases within computer vision, the adversarial examples are usually images formed by making small perturbations to an example image through a gradient-based optimization approach.  

![Panda Example](/Panda_example.png)
###### **Figure 1:** An adversarial input, overlaid on a typical image, can cause a classifier to miscategorize a panda as a gibbon. (Goodfellow et al. 2015) ######

In the context of object recognition, adversarial examples are formed by applying a small perturbation to an image from the dataset such that the ML classifier is coerced into misclassifying the image according to the requirement of the adversary. Figure 1 depicts a canonical example in which a relatively small perturbation to an image of a panda causes it to be misclassified as a gibbon. According to Elsayed et. al. (2018), “this perturbation is small enough to be imperceptible (i.e., it cannot be saved in a standard png file that uses 8 bits because the perturbation is smaller than 1/255 of the pixel dynamic range). This perturbation relies on carefully chosen structure based on the parameters of the neural network—but when magnified to be perceptible, human observers cannot recognize any meaningful structure.” This example depicts a counter-intuitive notion in which humans recognize the correct classification due to a high-level examination of the image, but a classifier is pushed towards misclassification due to its ability (or “unfortunate” ability in this case) to analyze the image in multiple dimensions by creating an effective latent space. 

As shown above, these adversarial examples could be dangerous. Hence, defensive strategies must be developed. In this regard, two strategies are widely used — brute force and defensive distillation (Examined in Section VI). Effectively, defensive distillation is used “to smooth the model learned by a DNN architecture during training by helping the model generalize better to samples outside of its training dataset.” (Papernot, et. al. 2016) Due to the number of permutations of the types of attacks, defensive tactics haven’t proven their merit as of yet. We believe that a specialized approach towards defensive strategies must be adopted wherein the characteristics of each network architecture must be taken into account in order to create effective and robust models. In other words, the defensive game requires us not to go wide in our approach (generalize defensive tactics), but go deep (within each architecture).

## III. Methodology:
With an aim to understand the robustness of various classification networks, we chose two widely used ones –- a Convolutional Neural Network (CNN) and a Variational Auto-Encoder (VAE). After separately training these networks on the Labeled Faces in the Wild (LFW) dataset and MNIST dataset, we deployed three different attacks on each dataset. These attacks are targeted attacks as their objective is to misclassify an image to a specified false label, which is different from the true class.

1. **Metrics used to understand whether the attack is “effective”:** To understand the vulnerability of each network, we define two criteria, validity and confidence, which end up assigning a degree of effectiveness to each attack. As an adversarial attack modifies the original image, an adversarial image is deemed as invalid if the image is distorted to the naked eye. On the other hand, each attack attempts to modify the image and classify it as a specific false label with some amount of confidence. This measure of confidence by which the network classifies an image as a false label is the second criteria. In the MNIST dataset, if the CNN classified the digit 5 as the digit 9 with a probability of 0.89 and showed no visible distortion then the attack was valid and successful with a confidence of 89%. The table below displays the four levels of effectiveness of an attack:

1. **Networks Used:**
    1. While most classification models based on neural networks are susceptible to adversaries, we are focussing on image classification models to study the cause and effect of such adversarial attacks. We examine adversarial attacks on Convolution Neural Networks (CNN) since they constitute the base of a majority of Industry standard image classification models. A simple CNN is prone to a large number of attacks and it becomes essential for us to consider a more foolproof architecture for image classification. 
    1. In comparison with a CNN, we expect the Variational Autoencoders to be more robust since it approximates the probability distribution of each feature of the input image. We suspect that using the output of the encoder as the input to the image classification model could fare better against attacks that are inconspicuous to the human eye. 
    1. *H<sub>0</sub> : In comparison with a CNN, we expect the Variational Autoencoders to be less vulnerable to adversarial attacks*
    
1. **Datasets Used :** Since we are concerned about the distinction between two different network architecture, we consider two contrasting datasets in terms of size and pixel density which could elucidate the differences between the two.

    1. First, we consider the most commonly used dataset to test Image Classification — MNIST. Experiments run on the MNIST dataset give us a basic idea of how a model would react to a minimal set of features. We must keep in mind that models trained on the MNIST dataset would have very high accuracy and this could affect it’s susceptibility to adversarial attacks. The striking features of the dataset which could provide insights are that each instance is a small 28 x 28 image in grayscale with a very low pixel density. 

    1. While MNIST is sufficient to run our pilot experimental studies, we approach our problem from the real world perspective to obtain more conclusive results. To this end, we use the Labelled Faces in the Wild (LFW) dataset which is a database of face photographs designed for studying the problem of unconstrained face recognition. It is vital to examine the robustness of network architectures that are used for facial authentication since the risk of failure is far greater than classifying handwritten numbers. In contrast to the MNIST dataset, LFW provides larger 154 x 154 images that have a high pixel density. Also, since the images are coloured, it provides adversaries more opportunities to manipulate images that can seem genuine to the human eye.

    1. *H<sub>1</sub> : Attacks on networks with high pixel density inputs are more effective relative to low pixel density attacks*
    
1. **Attacks Used:** The three attacks we deployed are white box attacks i.e. the attacker has complete information about the model’s parameters, gradient and so on. The following are the three gradient-based white box attacks:

    1. **L-BFGS:** The Limited-memory BFGS attack is a space-efficient version of the BFGS algorithm. This attack solves a simple optimization problem to misclassify an image while maintaining similarity between the original image and the adversarial image. In other words, given an image x, the algorithm aims to find an image y such that the L-2 distance between x and y is minimum and y is labelled differently by the network (Carlini et al, 2016).
    1. **Basic Iterative Attack:** This attack is a purely gradient-based attack where a small value, α, is multiplied with the gradient in the direction of the targeted class. This amount is clipped with another small value, , and added to each pixel value for multiple iterations until the image is falsely classified. Adding uniform noise in the direction of the targeted label leads the image to be falsely classified by the network. To ensure the similarity between the two images, the L-∞ distance is used.
    1. **L1 Basic Iterative Attack:** Identical to Basic Iterative Attack except the distance metric used is the L-1 distance.

## IV. Experimental Results:
The following Tables (2, 3) and Figures (2, 3, 4, 5) depict the results of our experiment on the VAE and CNN using the LFW and MNIST datasets.

![GitHub Logo](/images/logo.png)


## V. Inferences:
The following is a summary of our inferences from the above results. 

1. **The VAE is not vulnerable to low pixel density attacks.** This is because a low pixel input image yields a low confidence attack that is valid, resulting in a low degree of effectiveness of the attack.

1. **The VAE is vulnerable to high pixel density attacks.** This is because a high pixel image yields a high confidence attack that is invalid, resulting in a moderate degree of effectiveness of the attack.

1. **The CNN is highly vulnerable to high pixel density attacks.** This is because a high pixel image yields a high confidence attack that is valid, resulting in a high degree of effectiveness of the attack.

1. **The CNN is vulnerable to low pixel density attacks.** This is because a low pixel image yields a high confidence attack that is invalid, resulting in a moderate degree of effectiveness of the attack. 

In reference to the first hypothesis (H<sub>0</sub>) we observe that it holds true since the VAE is overall less susceptible to effective attacks given both MNIST and LFW datasets. On the other hand, the CNN is relatively vulnerable to the same attacks. This is expected since the VAE is creating a probability distribution for each feature within its latent space and therefore, should account for variations in images during classifications. 

In reference to the second hypothesis (H<sub>1</sub>), we observe that it is “somewhat” true. The reason for this indecision is because the pixel density alone cannot determine the effectiveness of the attack. It is also dependent on the model architecture. We can say for certain that a low pixel density image would yield a low validity or invalid adversarial image, but the degree of effectiveness (between moderate,  low and ineffective) is a function of the model. 

## VI. Defensive Strategies
The following are some existing defensive tactics to protect deep ML models from adversarial attacks along with our recommendations.

1. **Input Based:** Through our experiments, we observed that low-pixel density input images are highly effective in reducing the validity of the adversarial image. Hence, models which use black and white or/and small-sized input images will be relatively more secure against gradient based attacks. This follows from the intuition that low-pixel density images can be perturbed only negligibly due to the few numbers of pixels. Hence, any large perturbation will lead to the image looking different, which can be caught by the human eye.

1. **Procedural:** 

    1. Since the most effective white box attacks are gradient-based attacks, a major way to defend against these attacks are to hide the gradient from the attacker. Thus, by converting the setting of the attack to a black box, it is possible to defend against gradient-based attacks.
    1. Another widespread method used to defend against adversarial examples is defensive distillation. Defensive distillation has two main processes. The first process referred to as the teacher model is training the network in the standard manner. Then, by modifying the softmax function, we assign soft labels to each training example in the training model. For example, a soft label for an image of the digit 7 might say it has an 80% chance of being a seven and 20% chance of being a one. The soft labels obtained from the teacher model contain significantly more relevant information about the features of the image. So, by training over these soft labels, the distilled model learns the feature space for each image. This refined model becomes more robust as its vulnerability to an attacker trying to misclassify an image drastically reduces (Yuan et al). In our earlier example, if the same image of the digit 7 were to be attacked and perturbed to be classified as the digit 4, the distilled model would classify it as the digit 4 with extremely low accuracy. This is because of the difference between the feature space for the digit 7 and the digit 4.
    1. The last procedural defensive strategy we could adopt is a brute force strategy. By training the model overall adversarial examples, the effectiveness of the attack will inevitably reduce. However, the brute force strategy proves to be highly impractical because of the numerous types of attacks generating a high number of adversarial examples.
    
## VII. Future Scope and Conclusion
We have seen how different networks have fared against multiple types of attacks and concluded that each network has a vulnerability of its own. The next step in this field of study would be to secure these network using either of the defensive strategies mentioned above and further compare them. Given time and computation power, we will be able to train our CNNs with Defensive Distillation making it more robust.












