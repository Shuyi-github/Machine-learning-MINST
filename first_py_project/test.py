import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg


#f=open('')
trainset=np.loadtxt('digits4000_trainset.txt')
digits_vec=np.loadtxt('digits4000_digits_vec.txt')
lables=pd.read_csv('/Users/tangshuyi/Downloads/textbooks & lecture notes/machine learning/ml_project/digits4000_digits_labels.csv')
#print(data.head)
print("hello,python")
digits=np.asmatrix(digits_vec)
print(trainset[0])
i=1
img=digits[0]
img=img.reshape((28,28))
print(img)
plt.imshow(img,cmap='Greys')
#plt.show()

r=trainset[:,1]
print(r)

i=(int)(trainset[0][1])
print(i)
img1=digits[i]
img1=img1.reshape((28,28))
plt.imshow(img1,cmap='Greys')
#plt.show()
#img=np.matrix(digits_vec);
#print(img);
#img=img.reshape((28,28))
#plt.imshow(img,cmap='gray')
#plt.title(labels.iloc[i,0])