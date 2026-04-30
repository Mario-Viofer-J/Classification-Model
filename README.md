# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Load the Iris dataset using a suitable library.

### STEP 2:
Preprocess the data by handling missing values and normalizing features.

### STEP 3: 
Split the dataset into training and testing sets.

### STEP 4: 
Train a classification model using the training data.

### STEP 5: 
Evaluate the model on the test data and calculate accuracy.

### STEP 6: 
Display the test accuracy, confusion matrix, and classification report.




## PROGRAM

### Name: Mario Viofer J

### Register Number: 212223100032

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
df['target']=y
df.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)
train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)
train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
test_loader=DataLoader(test_data,batch_size=16)
class IrisClassifier(nn.Module):
  def __init__(self,input_size):
    super(IrisClassifier,self).__init__()
    self.fc1=nn.Linear(input_size,16)
    self.fc2=nn.Linear(16,8)
    self.fc3=nn.Linear(8,3)

  def forward(self,x):
    x= F.relu(self.fc1(x))
    x= F.relu(self.fc2(x))
    return self.fc3(x)
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
    if(epoch+1)%10==0:
      print(f'Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}')
model=IrisClassifier(input_size=X_train.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
train_model(model,train_loader,criterion,optimizer,epochs=100)
model.eval()
predictions,actuals=[],[]
with torch.no_grad():
  for X_batch,y_batch in test_loader:
    outputs=model(X_batch)
    _,predicted=torch.max(outputs,1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())
accuracy=accuracy_score(actuals,predictions)
conf_matrix=confusion_matrix(actuals,predictions)
class_report=classification_report(actuals,predictions,target_names=iris.target_names)
print(f'Test Accuracy: {accuracy:.2f}\n')
print("Classification Report:\n",class_report)
print("Confusion Matrix:\n",conf_matrix)
sns.heatmap(conf_matrix,annot=True,cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
sample_input=X_test[5].unsqueeze(0)
with torch.no_grad():
  output=model(sample_input)
  predicted_class_index=torch.argmax(output[0]).item()
  predicted_class_label=iris.target_names[predicted_class_index]
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')

```

### Dataset Information
<img width="560" height="176" alt="image" src="https://github.com/user-attachments/assets/2edbb5cf-377b-446e-81f2-73be9d66f279" />
<img width="573" height="190" alt="image" src="https://github.com/user-attachments/assets/6678614f-621d-4ff2-aa54-3be96c6aa68a" />
<img width="224" height="175" alt="image" src="https://github.com/user-attachments/assets/254f6f73-8b38-4fe6-8b8d-87dab8da9032" />

### OUTPUT

## Confusion Matrix

<img width="555" height="518" alt="image" src="https://github.com/user-attachments/assets/dd75baa8-4ced-4a7f-be56-8a224c64a353" />

## Classification Report
<img width="405" height="215" alt="image" src="https://github.com/user-attachments/assets/e3cb7d25-1884-462b-b855-39e88104147e" />

### New Sample Data Prediction
<img width="370" height="42" alt="image" src="https://github.com/user-attachments/assets/5901e363-0583-45ef-800c-b8f8695113d3" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch
