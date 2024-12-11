# chest_xray

Just homework ))






Multimedia Computing

Chest X-ray pneumonia detection
Custom CNN vs ResNet50 


 
Introduction
The main task of this report is to compare the difference in precision of custom created from scratch model with pre-trained big models. For this task the theme was chosen pneumonia diagnosis, with chest x-ray images and watch how would these 2 models behave. All code is on git hub repository and can be gained via QR on the last page.
Dataset
The dataset was found from Kaggle , it had a very good size for training, something like 
Loaded 5116 images for train split
Loaded 116 images for val split
The link I will provide even with QR )
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
 
Training custom CNN
I had 2 attempts to train very good model, the first one showed very good train f1 score, but poor val f1, it was kind of overfitting very.
Problem: High overfitting
- Train F1: 0.97
- Val F1: 0.56
- Difference: ~40%
I have used google colab as engine, 
My hyper parameters were 
batch_size = 128  
num_epochs = 15
learning_rate = 3e-4
In order to how to solve the problem with overfitting I analyzed, and it seems it was because critical small size of validation dataset (16 images), so I enlarged the validation dataset by randomly picking images from train dataset up to 100 images, enhanced regularization and added augmentation of data.
So the result was as expected very very good )) 
- Train F1: 0.86
- Val F1: 0.87
- Разрыв: ~1%

Training ResNet50
Turning to ResNet50, this was very smooth as I had already functions to lead dataset, to preprocess them, so and as it had millions of parameters, so I gave him 10 epochs. ( for balance)
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
This model showed very from the beginning and the overall results were:
- Val F1: 0.93 (only +6%)
- Size of model is bigger for  8.4 times
To be honest I expected very big and radical difference, but as we can see difference is not that big.
Comparison
 
![image](https://github.com/user-attachments/assets/ce604bb6-a271-49b3-a638-06d7b2eddc48)

Detailed Comparison:
Metric               | Custom CNN | ResNet50
Validation F1    |  0.870             | 0.930
Training F1       |  0.860             | 0.950
Validation Loss |  0.390             | 0.310
Training Loss          | 0.410  | 0.280
Parameters (M)        | 2.8     | 23.5
Epochs to Converge | ~8     | ~10

Confusion matrices
![image](https://github.com/user-attachments/assets/2556f9e8-ce0b-4fb6-ad1d-94cbe665a646)

 
ROC curves 
![image](https://github.com/user-attachments/assets/117cf59b-f461-4723-93ea-aa4be2818f8a)
 



Precision-Recall curves
 ![image](https://github.com/user-attachments/assets/934da049-b27b-4145-9b7f-f3a1f7bb5cd2)

Detailed classification reports:
Custom CNN Classification Report:
                     precision    recall    f1-score   support
Normal       | 0.9032      0.8235     0.8615        34
Pneumonia |  0.9294     0.9634     0.9461        82
accuracy      |                  	       0.9224       116
macro avg    |  0.9163    0.8935    0.9038       116
weighted avg|  0.9217    0.9224    0.9213       116

ResNet50 Classification Report:
                       precision    recall    f1-score   support
Normal         | 0.9259      0.7353    0.8197        34
Pneumonia   | 0.8989      0.9756    0.9357        82
accuracy       |                                 0.9052       116
macro avg     | 0.9124    0.8555     0.8777       116
weighted avg | 0.9068    0.9052    0.9017       116

Conclusion
So, in conclusion, sometimes even simple solutions like enlarging validation set would make very big impact than complicated architecture solutions. Three things I have learnt with this report, 
1) Quality of input data is more important than complexity of model, 
2) Correct validation is critical for evaluating the model,
3) Transfer learning is not always able to make dramatic improvements.
So, overall if I will have more time I would choose to upper my own model than taking others, as it takes less size, I can see what’s wrong always and play with hyper parameters.
Link to github
