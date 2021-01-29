from PIL import Image
import numpy as np
import gzip
import io
import os

PATH_TO_FASHION_MNIST = 'C:\\Users\\Jay Patel\\OneDrive\\Desktop\\ImageTesting'

train_idx3 = os.path.join(PATH_TO_FASHION_MNIST, 'train-images-idx3-ubyte (1).gz')
x_train= gzip.open(train_idx3)

train_idx1=os.path.join(PATH_TO_FASHION_MNIST, 'train-labels-idx1-ubyte (1).gz')
y_train=gzip.open(train_idx1)


test_idx3=os.path.join(PATH_TO_FASHION_MNIST, 't10k-images-idx3-ubyte (1).gz')
x_test=gzip.open(test_idx3)


test_idx1=os.path.join(PATH_TO_FASHION_MNIST, 't10k-labels-idx1-ubyte (1).gz')
y_test=gzip.open(test_idx1)

def convert_outside_image(img):
    pixel_matrix=np.asarray(img)
    new_img=[]
    for e in pixel_matrix:
        
        new_img+=list(e)
    return [new_img]

def byte_to_int(n):
    return int.from_bytes(n,'big')

def getimgarr(f,max_img,oustideIMG=False):

    if  oustideIMG:
        return convert_outside_image(f)  #Outside images need to be converted to pixel arrays seperately b/c of formatting
    
    _=f.read(4)
    total_images=byte_to_int(f.read(4))
    row=byte_to_int(f.read(4))
    col=byte_to_int(f.read(4))
   

    images=[]
    for i in range(min(total_images,max_img)):
        curr_img=[]
        for j in range(row*col):
           
            single_pixel=byte_to_int(f.read(1))
            curr_img.append(single_pixel)
        
        images.append(curr_img)
    return images
      

def calc_distances(x_train,curr_img):
    distances=[]
    
    for i in range(len(x_train)):
        sample=x_train[i]
        dist=0
        for j in range(len(sample)):
            x,y=sample[j],curr_img[j]
            dist+=(x-y)**2
        dist=(dist)**0.5
        distances.append([i,dist]) #Find distance of each test sample from the current image, with its index
    return distances



def knn(x_train,curr_img,y_train,k):
    
    distances=calc_distances(x_train,curr_img)

    distances.sort(key=lambda x: x[1])
    possible_answers={}
    for j in range(k):
        idx,dist=distances[j]

        ans=y_train[idx]
        if ans  not in possible_answers:
            possible_answers[ans]=0
        possible_answers[ans]+=1
    ans=max(possible_answers,key=possible_answers.get)                                 
    return ans



                 
def getlabelarr(y_train,max_labels,outsideLABEL=False):

    _=y_train.read(4)
    total_labels=byte_to_int(y_train.read(4))
    
    labels=[]
    for i in range(min(max_labels,total_labels)):
        
        labels.append((int.from_bytes(y_train.read(1),'big')))
            
    return labels

def main(x_train,y_train,x_test,y_test):

    x_train=getimgarr(x_train,60000)
    y_train=getlabelarr(y_train,60000)
    x_test=getimgarr(x_test,10000,False) #Last input is for outside images
    y_test=getlabelarr(y_test,10000) 
    # If it is an outside image, manually insert a list of labels
    articles={  0: "T-shirt/top",
                1:	"Trouser",
                2:	"Pullover",
                3:"Dress",
                4:	"Coat",
                5:	"Sandal",
                6:	"Shirt",
                7:	"Sneaker",
                8:	"Bag",
                9:	"Ankle boot" }
    total,correct=0,0
    for i in range(len(x_test)):
        img=x_test[i]
        my_ans=knn(x_train,img,y_train,191)
        actual_ans=y_test[i]
        if actual_ans==my_ans:
            correct+=1 
        total+=1
        print("My Guess: " + articles[my_ans] + ", Label: "+ articles[actual_ans]+ ", Accuracy: "+ str(correct/total*100))

main(x_train,y_train,x_test,y_test)









    





