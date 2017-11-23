import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg


#f=open('')
trainset=pd.read_csv('/Users/tangshuyi/Downloads/textbooks & lecture notes/machine learning/ml_project/digits4000_trainset.csv')
digits_vec=pd.read_csv('/Users/tangshuyi/Downloads/textbooks & lecture notes/machine learning/ml_project/digits4000_digits_vec.csv')
lables=pd.read_csv('/Users/tangshuyi/Downloads/textbooks & lecture notes/machine learning/ml_project/digits4000_digits_labels.csv')
#print(data.head)
print("hello,python")

i=1
img=np.matrix(digits_vec);
print(img);
img=img.reshape((28,28))
#plt.imshow(img,cmap='gray')
#plt.title(labels.iloc[i,0])