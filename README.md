# Purpose
This code trains a Convolutional Neural Net (CNN) to calculate the molecular formula of a chemical structure from an image.  In the short term, this is a useful proof of concept that a CNN can correctly represent features of a molecule from a rendering of its structure.

# Built on
- keras (theano)
- skimage

# Representations
The input to the model is a 180x180 image created by the NCATS renderer from a molfile.  The data the model is trained and tested on is from the NIH Molecular Libraries Small Molecule Repository. 54,000 molecules are analyzed with a 90/10 train/test split.  The molecular formula is represented in a 16-D vector of atom counts, one element for each atom that occurs in our dataset.

# Validation
Without cross-validation to optimize the model, we achieve 0.36 RMSE after 8 epochs.

