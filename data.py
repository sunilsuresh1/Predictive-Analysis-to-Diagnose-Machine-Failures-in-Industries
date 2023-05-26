import pandas as pd
import numpy as np
#Getting Data
df = pd.read_excel(r'D:/All Documents/TH.xlsx')
yval = pd.read_excel(r'D:/All Documents/Y.xlsx')
m=len(df)
df.insert(0, "bias", 1)
yval["Failure"].replace({"No": "0", "Yes": "1"}, inplace=True)
ones=(np.ones(m))
ones=(ones.reshape(8784,1))
print(yval) 
theta=np.array([[0],[0],[0]]);
alpha=0.01

#Sigmoid Function
sigval=df.dot(theta)
sigmoid=1/(1+(np.exp(-sigval)))
print(sigmoid)

ones=ones.astype(int)
yval=yval.astype(int)
y1=ones-yval
print(y1)

#Loss Function 
val=np.transpose((np.log(sigmoid))).dot(yval)+np.transpose((np.log(np.array([1])-sigmoid))).dot(y1);
cost=-1/m*val;
print("Loss Function Value is\n", cost)
print("Please Wait...")

#Gradient Descent 
for i in range (1,1000):
    sigval=df.dot(theta);
    sigmoid=1/(1+(np.exp(-sigval)));
   
    sigmoid=sigmoid.astype(int);
    yval=yval.astype(int);
    
    df1=(df.iloc[0:8784,0]);
    df2=(df.iloc[0:8784,1]);
    df3=(df.iloc[0:8784,2]);

    num1=df1.values
    num2=df2.values
    num3=df3.values
    
    arr1=num1.reshape((8784,1))
    arr2=num2.reshape((8784,1))
    arr3=num3.reshape((8784,1))

    sig1=sigmoid.values
    yval1=yval.values
    sig=(sig1-yval1)

    val1=np.multiply(sig,arr1);
    val2=np.multiply(sig,arr2);
    val3=np.multiply(sig,arr3);
    
    t1=(alpha/m*(sum(val1)));
    t2=(alpha/m*(sum(val2)));
    t3=(alpha/m*(sum(val3)));

#t1=alpha/m*sum((np.transpose((1/(1+np.exp(-(np.transpose(theta).dot(np.transpose(df)))))))-yval)*((np.transpose(df[:,0]).reshape(8784,1))))
#t2=alpha/m*sum((np.transpose((1/(1+np.exp(-(np.transpose(theta).dot(np.transpose(df)))))))-yval)*((np.transpose(df[:,1]).reshape(8784,1))))
#t3=alpha/m*sum((np.transpose((1/(1+np.exp(-(np.transpose(theta).dot(np.transpose(df)))))))-yval)*((np.transpose(df[:,2]).reshape(8784,1))))
    thet1=theta[0]-t1
    thet2=theta[1]-t2
    thet3=theta[2]-t3
    theta=np.array(([thet1,thet2,thet3]))

print ("Theta Values after Gradient Descent(Classifier) are \n", theta)
#theta=np.array([[-0.09220389],[0.64461885],[-0.68283197]])

#Test case 1
print ("test casel inputs 63,85")
x=np.array([1,63,85]);   
c=np.transpose(x).dot(theta);
sigmoid1=1/(1+(np.exp(-c)));
print(sigmoid1)
if sigmoid1 >= 0.5:
      print(1);
else:
      print(0);

#Test case 2
print ("test case2 inputs 71,65")
x=np.array([1,71,65]);   
c=np.transpose(x).dot(theta);
sigmoid1=1/(1+(np.exp(-c)));
print(sigmoid1)
if sigmoid1 >= 0.5:
      print(1);
else:
      print(0);
      
#Test case 3
print ("test case2 inputs 60,84")
x=np.array([1,60,84]);   
c=np.transpose(x).dot(theta);
sigmoid1=1/(1+(np.exp(-c)));
print(sigmoid1)
if sigmoid1 >= 0.5:
      print(1);
else:
      print(0);
