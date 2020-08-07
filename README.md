# Text-Independent-Writer-Identification-using-Convolutional-Neural-Network
1. My Internship work at Indian Statistical Institute Kolkata under the guidance of Prof. Umapada Pal(Head CVPR unit).
2. Implementation is done using keras deep learning framework. 

## Data Description 
1. The dataset consists of 228 handwritten documents written by 57 writers in Kannada Language
where each writer has written 4 documents.
2. All the input images are binary. (Values : 0 and 1)

## Data Preprocessing : 
1. Out of 4 Documents, 3 Documents written by each writer is used training, 1 document for testing. 
2. Extracting patches of size : (128, 128)
3. Generating the one hot encoded labels for all 57 writers.

## Training Details 
There are two phases. Model is trained for 30 epochs in first phase and learning rate is reduced, then trained for more 20 epochs keeping all other hyperparameters constant. 
1. Batch size : 32
2. Validation split : 0.15

### 1st phase : 
1. Learning rate : 0.0001
2. Epochs : 30

### 2nd phase : 
1. Learning rate : 0.000001
2. Epochs : 30 --> 50 

## Results : 

Accuracy : 76.12 %.

# Thank you
