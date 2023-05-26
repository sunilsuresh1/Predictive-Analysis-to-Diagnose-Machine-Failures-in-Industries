from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
df = pd.read_excel(r'D:/All Documents/TH.xlsx')
yval = pd.read_excel(r'D:/All Documents/Y.xlsx')
m=len(df)
df.insert(0, "bias", 1)
yval["Failure"].replace({"No": "0", "Yes": "1"}, inplace=True)
yval=yval.astype(int);
#99.32%
#theta=np.array([[-0.09220389],[-0.64461885],[-0.68283197]])
#for 1000 iterations sum outside
#theta=np.array([[ 0.00823201],[ 0.54651753],[-0.01871926]])
#for 10000 iterations sum outside
#theta=np.array([[ 0.02068306],[ 0.54915073],[-0.01488957]])
#theta=np.array([[ 1],[ 0.54915073],[-0.01488957]])


#theta=np.array([[0.00397951],[0.29071915],[0.18443909]])
#theta=np.array([[1],[0.29071915],[0.18443909]])
#theta=np.array([[ 0.0091041 ], [ 0.77256175], [-0.24687753]])


#theta=np.array([[ 1 ], [ 0.77256175], [-0.24687753]])
#theta=np.array([[0.0108587 ],[0.51926517],[0.01062015]])
#99% accuracy
#theta=np.array([[ -0.92203893],[-64.51995902],[-68.34471311]])
#theta=np.array([[ 1],[-64.51995902],[-68.34471311]])


#recent
#theta=([[0.00711942], [0.49678518], [0.03161066]])
#theta=([[0.00397951], [0.29071915], [0.18443909]])
#theta=([[ 0.00863604], [ 0.53879804], [-0.00310599]])
#theta=([[0.00092121], [0.06446189], [0.0682832 ]])

#theta=([[0.0056431 ], [0.41843044], [0.08964162]])

print(df)
print(yval)
c=df.dot(theta);
print(c)
sigmoid1=1/(1+(np.exp(-c)));

sig=sigmoid1.values

for i in range(1,8784):
    if sig[i] >= 0.5:
        sig[i]=1
    else:
        sig[i]=0
        
y_pred=sig.astype(int)
#print(confusion_matrix(yval, y_pred))

tn,fp,fn,tp=confusion_matrix(yval,y_pred).ravel()
print("Evaluation Values are...")

print("True Negatives:",tn)
print("False Positives:",fp)
print("False Negatives:",fn)
print("True Positives:",tp)

print("Accuracy value:",end="")
print("{0:.2f}%".format(100*accuracy_score(yval,y_pred)))
print("Precision Score:",precision_score(yval,y_pred))
print("Recall Score:",recall_score(yval,y_pred))
print("F1 Score:",f1_score(yval,y_pred))
#print(y_pred)
print(y_pred.shape)
#y_pred.head(155)

