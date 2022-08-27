# image_classifier

AWS AI & ML programming with Python Nanodegree .

Install all the packages mentioned in the requirements.txt file using : ```pip install -r requirements.txt```

### Basics : The foundations for building your own neural network
Pyhton, Numpy, Pandas, Matplotlib, PyTorch, and Linear Algebra

### project1 : Use a pre-trained image classifier to identify dog breeds

PURPOSE: Classifies pet images using a pretrained CNN model, compares these classifications to the true identity of the pets in the images, and summarizes how well the CNN performed on the image classification task.

With this program we will be comparing the performance of 3 different CNN model architectures {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16},  to determine which provides the 'best' classification.

### NumPy Mini-Project : Mean_Normalization_and_Data_Separation.ipynb

In machine learning we use large amounts of data to train our models. Some machine learning algorithms may require that the data is normalized (range of values be between 0 and 1) in order to work correctly.

After the data has been mean normalized, it is customary in machine learnig to split our dataset into three sets (chosen at random, making sure that we don't pick the same row twice) :
1. A Training Set : contains 60% of the data
2. A Cross Validation Set : contains 20% of the data
3. A Test Set : contains 20% of the data
