import sys
import cv2
import numpy as np


img1=cv2.imread(sys.argv[1])
img1=cv2.resize(img1, (900,500))
img2=cv2.imread(sys.argv[2])
img2=cv2.resize(img2, (900,500))
layer1=img1.copy()
layer2=img2.copy()



def gaussian_pyramid(img,n):
   gp=[img]    #gaussian pyramidi array first value is image
   for i in range(n): #5 time going to resolution.   n-1
        layer=cv2.pyrDown(gp[i])
        gp.append(layer)
        #cv2.imshow(str(i),layer)
   return  gp



def laplacian_pyramid(gp,n):
    lp=[gp[n]]
    for i in range(n,0,-1):
        size_img1=(gp[i-1].shape[1],gp[i-1].shape[0])
        gaussian_extended= cv2.pyrUp(gp[i], dstsize=size_img1) #upper level of gaussian pyramid
        laplacian=cv2.subtract(gp[i-1], gaussian_extended)
       # cv2.imshow(str(i),laplacian)
        lp.append(laplacian)
    return lp


def blend_images(lp1,lp2,gaussian_mask):
    blended_images=[]
    for l1,l2,mask in zip(lp1,lp2,gaussian_mask):
        #cv2.imshow('masksss ', l1*mask)
        #cv2.imshow('masksss2 ',(1.0- mask)*l2)
        bi=l2*mask + l1*(1.0-mask)
        blended_images.append(bi)
    return blended_images


# now reconstruct
def reconstruct_image(blended_image,n):
    reconstruction= blended_image[0]
    reconstruction_list=[reconstruction]
    for i in range(n-1):#lr=cv2.pyrDown(img) #lr is low resolution
        size = (blended_image[i+1].shape[1], blended_image[i+1].shape[0])#lr2=cv2.pyrDown(lr) #lr is low resolution2
        reconstruction_expanded = cv2.pyrUp(reconstruction, dstsize = size) #up to sixth level
        reconstruction = cv2.add(blended_image[i+1],reconstruction_expanded)#hr=cv2.pyrUp(lr2) #hr is higher resolution
        reconstruction_list.append(reconstruction)
    return reconstruction_list




n_gaussian=6
n_laplacian=5
gaussian_pyramid_for_first_image=gaussian_pyramid(layer1,n_gaussian)   #gaussian pyramid fot both of image
gaussian_pyramid_for_second_image=gaussian_pyramid(layer2,n_gaussian)   #gaussian pyramid fot both of image
#cv2.imshow('gauss', gaussian_pyramid_for_first_image[1])
laplacian_pyramid_for_first_image=laplacian_pyramid(gaussian_pyramid_for_first_image,n_laplacian)          #laplacian pyramid fot both of image
laplacian_pyramid_for_second_image=laplacian_pyramid(gaussian_pyramid_for_second_image,n_laplacian)         #laplacian pyramid fot both of image

mask=np.zeros((500,900,3), dtype='float64')
mask[125:350,320:800,:]=(1,1,1)
#print(mask.shape)
#cv2.imshow('mask', mask )
gaussian_mask=gaussian_pyramid(mask,n_gaussian-1)

gaussian_mask.reverse()
#cv2.imshow('mask', gaussian_mask[1])

blend_images=blend_images(laplacian_pyramid_for_first_image,laplacian_pyramid_for_second_image,gaussian_mask)

reconstruct_image=reconstruct_image(blend_images,len(blend_images))
final_image=cv2.normalize(reconstruct_image[5].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('reconstructed image', final_image)



#cv2.imshow('original image ', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()