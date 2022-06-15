# PROBLEM STATEMENT
The goal of this project is to create a model that can diagnose TB from CXR and graphically convey the results using Deep Convolutional Neural Networks. Over the same data set, the performance of this suggested model is compared to that of preset CNNs. The project is being undertaken out utilising the National Institute of Health (NIH) of the United States' publicly available datasets Shenzhen and Montgomery.
VGG-168, VGG-198, ResNet-506, AlexNet4, and Xception9 are modified versions of pre-trained Convolutional Neural Networks models used in the first two rounds of the project. The first part of the project is to find the optimum pre-trained CNN for automatic Feature Extraction through experimentation. The purpose of the second step of the project is to verify the performance of the pre-trained CNN identified in the first stage. The study effort concludes with the proposal of a novel handcrafted CNN architecture for TB identification.
# AIM 
The main goal of this study is to propose a deep neural network structure, especially a Convolutional Neural Network (CNN) model, for dynamically classifying and identifying CXR with active and latent TB infection. Recent work in the field of CXR – Classification have included pre-trained deep learning models such as AlexNet, GoogLeNet5, and ResNet, but these models were designed to classify real pictures. These classifiers are extremely strong, having been designed and refined to distinguish between hundreds of different classes and having been trained on massive amounts of data. As a result, they need a lot of memory and a lot of processing power.They have a high degree of freedom, which makes them prone to overfitting, and they do not adapt well when used for medical picture classification, which requires relatively little data.
# PROPOSED MODEL
The developed deep learning model is specifically built for diagnosing Tuberculosis from chest X-rays. To guarantee that the model learns to its full potential and does not compromise classification performance, augmentation and different pre-processing approaches have been used. It's also faster in terms of processing and memory than pre-trained deep learning models. The second half of the study focuses on visualising the findings for easier understanding. The gradient class activation mapping (Grad-CAM) approach was used to highlight the critical portions of the CXR pictures for the prediction procedure. These visualisations can also assist in the monitoring of the damaged lung areas as the disease progresses.
