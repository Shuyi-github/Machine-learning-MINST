import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler


#f=open('')
trainset=np.loadtxt('digits4000_trainset.txt')
digits_vec=np.loadtxt('digits4000_digits_vec.txt')
labels=np.loadtxt('digits4000_digits_labels.txt')
testset=np.loadtxt('digits4000_testset.txt')

print("hello,python")
digits=np.asmatrix(digits_vec)
#print(trainset[0])
i=1
img=digits[0]
img=img.reshape((28,28))
#print(img)
plt.imshow(img,cmap='Greys')
#plt.show()

train_index=trainset[:,1]    #all training data's index
#print(train_index)
test_index=testset[:,1]


test_data=[]
test_label=[]
for x in test_index:
    img=digits_vec[(int)(x-1)]
    #img=img.reshape((28,28))
    test_data.append(img)
    label=labels[(int)(x-1)]
    test_label.append((int)(label))

da=[]
train_label=[]
for x in train_index:
    img=digits_vec[(int)(x-1)]
    #img=img.reshape((28,28))
    cols, = img.shape
    for j in range(cols):
            if (img[j] <= 128):
                img[j] = 0
            else:
                img[j] = 255

    da.append(img)
    label=labels[(int)(x-1)]
    train_label.append((int)(label))

i=(int)(trainset[1999][1])
#print(i)
img1=da[10]
img1=img1.reshape((28,28))
print('show...')
print(img1)
img2=da[100]
img2=img2.reshape((28,28))

print('.....')
print(img1)
#img1=plt.imshow(img1,cmap='Greys')
#plt.show()

plt.figure(1)
plt.subplot(211)
#plt.imshow(img1)

plt.subplot(212)
plt.imshow(img2)
#plt.show()
#d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
print(da[0])
print('pre-processing...')
#da = np.array(da)
#print(da)
train_data=da
trainwithoupca=da
testwithoutpca=test_data
pca = PCA(n_components=0.5, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)
print('after pca process..,')
print(train_data)
#print(train_data)


print('Train SVM...')
print(train_label)
svc = SVC(degree=10,kernel='rbf',C=5,gamma=0.05)
svc.fit(train_data, train_label)

print('Predicting...')
test_data = pca.transform(test_data)
predict = svc.predict(test_data)
print(predict)


col,=predict.shape
print(col)
j=0
for i in range(col):
    if(predict[i]==test_label[i]):
        j=j+1
    else:
        print('not match')

print(j)

print('evaluating...')
print()
acc=svc.score(test_data,test_label)
print('accuracy:',acc)

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(trainwithoupca, train_label)
print('KNN...')
test = clf.predict(testwithoutpca)
acc_KNN= clf.score(testwithoutpca,test_label)
print(acc_KNN)

