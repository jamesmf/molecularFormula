# Purpose
This code explores the utility of deep learning on images of chemicals.  There is obvious value in the field of chemical OCR as well as potential value in learning deep representations of chemicals for QSAR/QSPR.

The molecular OCR goals and methods are explained [here](https://frickj.wordpress.com/2016/01/19/deep-learning-for-molecular-ocr/#more-56).


#Chemical OCR
We explore the possibility of using Convolutional Neural Networks to aid rules-based chemical OCR by either providing a map of atom locations within the image or by providing an Extended Chemical FingerPrint (ECFP) like vector to properly bound the rules-based method.  An accurate feature vector would also make it very simple to find similar structures in an indexed database.

#Learning Descriptors
Chemicals can be hard to embed into an informative feature space for learning.  State of the art techniques include ECFPs and vectors of chemical descriptors.  Concurrent to this research, there has been a paper on attempting to learn features using CNNs on atom-level features summed over all neighborhoods in a molecule.  This image-based approach seeks to avoid the pitfalls of iterating over atoms by considering the entire molecule at once.  This approach demonstrates the ability to reconstruct  basic ECFP features and thereby shows it is capable of learning at least the same information as ECFP. However since the features this approach will learn will be driven by the prediction task, we argue it should extract more informative features.

# Built on
- scipy/numpy stack
- keras (theano)
- skimage

# Input
The input to the model is a 300x300 image created by the NCATS renderer from a molfile.  The data the model is trained and tested on is from the NIH Molecular Libraries Small Molecule Repository. 54,000 molecules are analyzed with a 90/10 train/test split.

#Output
We have explored learning different feature representations. For molecular OCR, we have defined an output space comprised of basic molecular units: atom counts, atom-atom bond counts(C-C, C-N, etc),bond type counts, and smallest-set-of-smallest-ring counts.


![Alt text](https://cloud.githubusercontent.com/assets/7809188/11154289/7ce6006a-8a0c-11e5-90ad-c572ee5f9b26.jpg)
