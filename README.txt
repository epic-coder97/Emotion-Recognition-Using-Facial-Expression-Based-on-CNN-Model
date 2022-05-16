---------------------------------------------------EMOTION DETECTION USING FACIAL EXPRESSIONS----------------------------
-------------------------------------------------------------DATA 602- FINAL PROJECT---------------------------


We have implemented 3 different Machine Learning models on the dataset to find the best performing model and deploy it in our application which 
classifies the emotions using 7 categories. After observing the performance of different models, we decided to implement CNN model as our main model for 
image classification. The train.py file in the folder makes use of the CNN model to perform classification
This file will help you understand how to run the .ipynb and GUI source code.

1. Running .ipynb files.

We have used google collab to run these algorithms on the dataset. Therefore, please open these .ipynb files in your Google Collab.
The dataset is uploaded on the Google Drive and shared with everyone in the project. Please follow steps mentioned below to access the data.

Go to your umbc account's google drive, go to shared with me,
find a folder with name: 'archive', right click, click on 'Add Shortcut to Drive', then click on 'My drive' and click on 'Add Shortcut' button.

Also make sure the GPU is enabled in the Google Collab. To enable the GPU please follow below mentioned steps:
Go to the notebook >> Click on 'EDIT' on the top >> Click on 'Notebook Settings' >> Click on 'Hardware Accelarator' and add GPU.

After the above step, please run .ipynb files. 


2. Running gui.py 

In order to run this program, please download the entire project folder. Open Anaconda prompt in this folder's location and use the following command to run the file.

Command: python gui.py

Also, please make sure all required python libraries are installed locally.


3. Retraining the model (optional)

If you wish to retrain the model, please Open Anaconda prompt in this folder's location and use the following command to train the model again.

Command: python train.py
