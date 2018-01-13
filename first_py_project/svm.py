import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler

trainset=np.loadtxt('digits4000_trainset.txt')
digits_vec=np.loadtxt('digits4000_digits_vec.txt')
labels=np.loadtxt('digits4000_digits_labels.txt')
testset=np.loadtxt('digits4000_testset.txt')

challengelabel=np.loadtxt('cdigits_digits_labels.txt')
challengedigits_vec=np.loadtxt('cdigits_digits_vec.txt')

print("hello,python")
digits=np.asmatrix(digits_vec)


train_index=trainset[:,1]    #all training data's index
#print(train_index)
test_index=testset[:,1]

test_data=[]
test_label=[]
for x in test_index:
    img=digits_vec[(int)(x-1)]
    test_data.append(img)
    label=labels[(int)(x-1)]
    test_label.append((int)(label))

train_data=[]
train_label=[]
for x in train_index:
    img=digits_vec[(int)(x-1)]
    train_data.append(img)
    label=labels[(int)(x-1)]
    train_label.append((int)(label))



print('pre-processing...')

#print(train_data)

train_data=np.asarray(train_data)
test_data=np.asarray(test_data)
challenge_data=np.asarray(challengedigits_vec)



plt.figure(3)
plt.hist(train_data[1])

#train_data[train_data>0]=1
#test_data[test_data>0]=1

trainwithoutpca=train_data
testwithoutpca=test_data

#challenge_data[challenge_data>0]=1

g=0.05
c=10
kernel='rbf'

pca = PCA(n_components=26, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)
print('after pca process..,')

print('Train SVM...')
print(train_label)
svc = SVC(kernel=kernel,C=c,gamma=g)
svc.fit(train_data, train_label)

print('Predicting...')
test_data = pca.transform(test_data)
predict = svc.predict(test_data)
#print(predict)

col,=predict.shape
#print(col)
failure=[]
failure_label=[]
true_label=[]
j=0
m=0
for i in range(col):
    if(predict[i]==test_label[i]):
        j=j+1
    else:
        failure.append(trainwithoutpca[i].reshape(28,28))
        failure_label.append(predict[i])
        true_label.append(test_label[i])



failure1=failure[21:30]
failure2=failure[31:40]
failure3=failure[71:80]
failure4=failure[81:90]
failure5=failure[41:50]
failure_image1 = np.hstack(failure1)
failure_image2 = np.hstack(failure2)
failure_image3 = np.hstack(failure3)
failure_image4 = np.hstack(failure4)
failure_image5 = np.hstack(failure5)
failure_image=np.vstack((failure_image1,failure_image2,failure_image3,failure_image4,failure_image5))
print(failure_image)
plt.figure(4)
plt.imshow(failure_image.T,cmap='binary')

print('matching tuples:',j)

print('evaluating...')
print('SVM...')
acc=svc.score(test_data,test_label)
print('accuracy:',acc)

challenge_data=pca.transform(challenge_data)
predict_c=svc.predict(challenge_data)
acc_challenge=svc.score(challenge_data,challengelabel)
print(predict_c)
print('challenge set accuracy',acc_challenge)




plt.figure(2)
plt.hist(trainwithoutpca[1])
plt.figure()
plt.margins(0.1)
plt.xticks(np.arange(11)+0.25, ('0', '1', '2', '3', '4','5','6','7','8','9'))
plt.hist(true_label, bins=range(0, 11, 1),alpha=0.5,width=0.5)
plt.title('misclassified labels')
plt.show()

d = {'true label': test_label, 'predict label': predict}
df = pd.DataFrame(d)
df.index.name='index'
df.index+=1
df.to_csv('results_svm.csv', header=True)

d = {'true label': true_label, 'failure label': failure_label}
df = pd.DataFrame(d)
df.index.name='index'
df.index+=1
df.to_csv('failure_svm.csv', header=True)

print('cross-validation...')
pca = PCA(n_components=26, whiten=True)
pca.fit(testwithoutpca)
testwithoutpca = pca.transform(testwithoutpca)

#print('Train SVM...')
#print(test_label)
svc = SVC(kernel=kernel,C=c,gamma=g)
svc.fit(testwithoutpca, test_label)

trainwithoutpca = pca.transform(trainwithoutpca)
predict = svc.predict(trainwithoutpca)
#print('evaluating...')
acc_cross=svc.score(trainwithoutpca,train_label)
print('accuracy:',acc_cross)
averge_acc=(acc_cross+acc)/2
print('average accuracy for svm classifier:',averge_acc)