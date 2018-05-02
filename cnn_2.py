import cv2
import glob
import time
import os
import numpy as np
import pandas as pd
from collections import Counter
im_size=48
images=[]
im_path="/home/pratik/IITB/SEMESTER_2/EE769/Project/MIASDB/"
#imcrop_path="/home/pratik/IITB/SEMESTER_2/EE769/Project/MIASDB_crop/"
imcrop_path="/home/pratik/IITB/SEMESTER_2/EE769/Project/MIASDB_crop_2/"
t=time.time()
datainfo=pd.read_csv("datainfo.csv")

X=datainfo['x']
Y=datainfo['y']
instances=datainfo['ref_num']
labels=datainfo['severity']
abn=datainfo['abnormality']
print(Counter(labels))
print(Counter(abn))

R=datainfo['radius']
instances_transformed=[]
labels_transformed=[]

for i in range(330):
    im=im_path+instances[i]+".pgm"
    lb=labels[i]
    x=X[i]
    y=Y[i]
    r=R[i]
    if x!=x or y!=y or r!=r:
        x=int(512)
        y=int(512)
        r=int(24)

    crop_width=int(2*r)
    crop_height=int(2*r)
    x=int(int(x)-crop_width/2)
    y=int(1024-int(y)-crop_height/2)

    image=cv2.imread(im)
    crop_image=image[y:(y+crop_width),x:(x+crop_height)]
    crop_image=cv2.resize(crop_image,(im_size,im_size))
    fnm=instances[i]+"_"+str(i+1)+".pgm"

    if lb=='N':#save normal images as it is
        cv2.imwrite(imcrop_path+fnm,crop_image)
        instances_transformed.append(instances[i]+'_'+str(i+1))
        labels_transformed.append(lb)
    else:#rotate and flip abnormal images
        M=cv2.getRotationMatrix2D((im_size/2,im_size/2),90,1.0)
        image_90=cv2.warpAffine(crop_image,M,(im_size,im_size))
        M=cv2.getRotationMatrix2D((im_size/2,im_size/2),180,1.0)
        image_180=cv2.warpAffine(crop_image,M,(im_size,im_size))
        M=cv2.getRotationMatrix2D((im_size/2,im_size/2),270,1.0)
        image_270=cv2.warpAffine(crop_image,M,(im_size,im_size))
        image_vflip=cv2.flip(crop_image,1)

        #image file names
        fnm_90=instances[i]+'_90_'+str(i+1)+".pgm"
        fnm_180=instances[i]+'_180_'+str(i+1)+".pgm"
#        fnm_270=instances[i]+'_270_'+str(i+1)+".pgm"
        fnm_vflip=instances[i]+'_vflip_'+str(i+1)+".pgm"

        #image names for csv file
        instances_transformed.append(instances[i]+'_'+str(i+1))
        instances_transformed.append(instances[i]+'_90_'+str(i+1))
        instances_transformed.append(instances[i]+'_180_'+str(i+1))
        instances_transformed.append(instances[i]+'_vflip_'+str(i+1))
        
        #labels for csv file
        for k in range(4):
            labels_transformed.append(lb)
        
        #save processed images
        cv2.imwrite(imcrop_path+fnm,crop_image)
        cv2.imwrite(imcrop_path+fnm_90,image_90)
        cv2.imwrite(imcrop_path+fnm_180,image_180)
#        cv2.imwrite(imcrop_path+fnm_270,image_270)
        cv2.imwrite(imcrop_path+fnm_vflip,image_vflip)


    print(i)
print(Counter(labels_transformed))
tranformed_data={'ref_num':instances_transformed,'severity':labels_transformed}
d=pd.DataFrame(data=tranformed_data)
d.to_csv('transformed_data_info_2.csv',index=False)
print('Execution time: ')
print(time.time()-t)

