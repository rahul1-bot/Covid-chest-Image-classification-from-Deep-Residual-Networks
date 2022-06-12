# _An Efficient Supervised Deep Learning Approach for Covid chest Image classification from Deep Residual Networks_

## _Abstract_
With the outbreak of the COVID-19 and its various mutations making the infections faster and severe, it is becoming extremely important to determine the presence of COVID-19 infection in one’s body at a faster pace. Tests of Molecular, Antigen and Chest Scans are conducted to determine the presence of infection in the body, however, the molecular and antigen tests like RT-PCR require some time ranging from 1-5 days depending upon the availability of lab in the locality and how they run their tests with equipment. On the other hand, Chest scans like X-Ray and CT scans require lesser time of 10-15 minutes for detection by MDs. But due to the rise in cases and increase in demand for tests, radiologists and MDs find it harder to respond in time. Chest X-Rays are preferred for their less intensity and effective cost compared to CT scans. The model presented in this paper has operated on a total of 317 images containing COVID-19, Viral Pneumonia and Normal Chest X-Ray images. The model achieves an accuracy of 99.5% in the testing phase for classification of a COVID-19 infected Chest X-Ray. The aim is to help reduce the time taken for identifying infected X-Rays thus helping conduct tests at a faster pace. 
_Helping Deep Learning and AI Enthusiasts like me to contribute to improving COVID-19 detection using just Chest X-rays._

## _Keywords_
1) Artificial Neural Networks
2) Convolutional Neural Networks 
3) Image and Pattern Recognition 
4) COVID-19 Classification. 

## _Methodology_
![image](https://user-images.githubusercontent.com/65220704/132106660-75869364-19cc-4a17-8a68-58939ba24bd9.png)

## _Sneek Peek at the Dataset_
![image](https://user-images.githubusercontent.com/65220704/132106704-abe7e22b-b12d-4889-ab0e-c096fea3b328.png)



## _Model Architecture_
![image](https://user-images.githubusercontent.com/65220704/132106720-2c68f29e-1c5e-4d6d-bd27-a0a054bfb20a.png)

![Original-ResNet-18-Architecture](https://user-images.githubusercontent.com/65220704/132106510-02b931d3-7e48-459f-8977-22dbce19ef79.png)

A residual network, or ResNet for short, is an artificial neural network that aids in the construction of deeper neural networks by employing skip connections or shortcuts to bypass some levels. You will see how skipping aids in the construction of deeper network layers while avoiding the issue of disappearing gradients. ResNet comes in a variety of versions, such as ResNet-18, ResNet-34, ResNet-50, and so on. Even though the design is the same, the numbers represent layers. In our work we used the ResNet 18 model. It contains 18 layers. The layers which are involved in different ResNet Model. First there is a conv layer of (7x7) filter, then 16 conv layers are used with filter size of (3x3) and at last average pooling is done then the output is passed through the fully connected layers and output from the fully connected layer is passed through SoftMax activation function.


## _Results_
![image](https://user-images.githubusercontent.com/65220704/132106741-ff78753b-3796-4355-84df-35d5b787b5f8.png)
![image](https://user-images.githubusercontent.com/65220704/132106749-96f631ca-0766-4d54-8d83-5920915cb08c.png)
![image](https://user-images.githubusercontent.com/65220704/132106872-2dd86ce3-5c84-4767-9349-bd4814f80bad.png)
![image](https://user-images.githubusercontent.com/65220704/132106781-31b4ee40-aa8f-4197-b178-3a3309de61df.png)

Assessing and approving the acceptance of the arranged methodology, we tend to direct this procedure multiple times by changing and adjusting the hyperparameters of the model. Each time different outcomes are acquired. This research paper shows the most appropriate result. The generated confusion matrixes for the training and validation datasets illustrates the efficient classification performance of the generated model. 
A Deep Neural Network is highly robust to the fluctuating environment which makes it immune to noise. Also, many hidden layers help the model to learn complex data patterns.  

From the above plots, we can see that after reaching 20 epochs, the model has reached its maximum and minimum attained accuracy and validation loss in the validation dataset. Thus, we can early stop our training loop after 20 epochs to avoid the unnecessary computation time.
Moreover, access to the tune hyperparameters of the network makes it highly domain-specific and results in the higher accuracy which is in the case of covid image classification came out to be 100% for the training dataset and 99.5% for the validation dataset.  

## _Dataset Link_
_https://www.kaggle.com/pranavraikokte/covid19-image-dataset_

## _Conclusion_
Undoubtedly, covid can be a fatal disease which can become very challenging to diagnose with high accuracy and precision. Therefore, creating this model would be highly beneficial in the clinical trials. Thus, we developed a CNN based deep neural network which observes and classify chest CT scans images into 3 classes. That CNN model begins by reconstructing frontal chest images into compressed size and classify them whether an individual is tainted with either covid, normal or contains viral pneumonia. This powerful model can be applied to various healthcare firms, which is a far better approach than any other traditional clinical trials. Nonetheless, there are lots of problems which must be solved in this astonishing field. In future, this architecture will exhibit various application in different domains as well.

## Published Paper Link
https://ieeexplore.ieee.org/document/9633472
