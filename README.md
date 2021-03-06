# PROBLEM STATEMENT
The goal of this project is to create a model that can diagnose TB from CXR and graphically convey the results using Deep Convolutional Neural Networks. Over the same data set, the performance of this suggested model is compared to that of preset CNNs. The project is being undertaken out utilising the National Institute of Health (NIH) of the United States' publicly available datasets Shenzhen and Montgomery.
VGG-168, VGG-198, ResNet-506, AlexNet4, and Xception9 are modified versions of pre-trained Convolutional Neural Networks models used in the first two rounds of the project. The first part of the project is to find the optimum pre-trained CNN for automatic Feature Extraction through experimentation. The purpose of the second step of the project is to verify the performance of the pre-trained CNN identified in the first stage. The study effort concludes with the proposal of a novel handcrafted CNN architecture for TB identification.
# AIM 
The main goal of this study is to propose a deep neural network structure, especially a Convolutional Neural Network (CNN) model, for dynamically classifying and identifying CXR with active and latent TB infection. Recent work in the field of CXR – Classification have included pre-trained deep learning models such as AlexNet, GoogLeNet5, and ResNet, but these models were designed to classify real pictures. These classifiers are extremely strong, having been designed and refined to distinguish between hundreds of different classes and having been trained on massive amounts of data. As a result, they need a lot of memory and a lot of processing power.They have a high degree of freedom, which makes them prone to overfitting, and they do not adapt well when used for medical picture classification, which requires relatively little data.
# PROPOSED MODEL
The developed deep learning model is specifically built for diagnosing Tuberculosis from chest X-rays. To guarantee that the model learns to its full potential and does not compromise classification performance, augmentation and different pre-processing approaches have been used. It's also faster in terms of processing and memory than pre-trained deep learning models. The second half of the study focuses on visualising the findings for easier understanding. The gradient class activation mapping (Grad-CAM) approach was used to highlight the critical portions of the CXR pictures for the prediction procedure. These visualisations can also assist in the monitoring of the damaged lung areas as the disease progresses.

![image](https://user-images.githubusercontent.com/104749585/174728571-1b300688-edfc-4fc7-8749-97f36c505494.png)





# Findings and Observations
On the proposed model, a 10-fold cross validation was performed on all data sets, Shenzhen and Montgomery datasets individually and after integrating both. In terms of accuracy and AUC, the suggested model outperforms the pre-trained models.
![image](https://user-images.githubusercontent.com/104749585/174725950-05036909-915b-4624-89d5-72243ed3bd86.png)
![image](https://user-images.githubusercontent.com/104749585/174726861-a4646c0c-0123-4379-ad37-ef48907590eb.png)
![image](https://user-images.githubusercontent.com/104749585/174726967-62319c85-4f04-470e-9b6b-9badc95d0108.png)

# Conclusion
This study provides a suggested model based on a handcrafted neural network architecture that has been specifically tuned for TB diagnosis. When compared to a tiny network like GoogLeNet, which has roughly 7 million parameters, the suggested model has the benefit of having just about 230,000 parameters for prediction. Other structures, which have been used in other investigations, include as many as 70 million parameters. As a result, we infer that the parameter efficiency of our custom suggested model is the highest. The only regularisation methods used are batch normalisation and data augmentation. Our model is less prone to overfitting since it is a compact network with fewer degrees of freedom.

The model was independently examined in terms of accuracy and AUC shown for the Shenzhen and Montgomery data sets separately, and it was determined that our model exceeds all previous known studies, as shown in Table. When the model was trained on the merged dataset, the accuracy and AUC were found to be considerably more difficult to achieve than the previous findings. The suggested model was discovered to be both a compact and a strong classifier.

![image](https://user-images.githubusercontent.com/104749585/174728228-e0bc6079-c2bb-4a94-b132-9e73d94ae374.png)


Tuberculosis detection is not just limited to X-ray examination, but also includes studying various other factors pertaining to a patient, such as the patient's health history, lab reports, tests, and so on, based on the inputs gathered from healthcare professionals who specialise in treating tuberculosis. These new data points will undoubtedly aid in the diagnosing procedure. Combining all of the sources of input and developing a model that uses clinical inputs as well as picture data would be an intriguing area of research.

