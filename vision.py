from colorsys import rgb_to_hls, rgb_to_hsv
from sqlite3 import Timestamp
import cv2 as cv
import numpy as np
import ctypes
from datetime import datetime
from matplotlib import pyplot as plt
from skimage.util import random_noise


## get Screen Size
user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
 
def readImage(imageName: cv.Mat = "Lag"):
    
    imgReturn = cv.imread("Images/" + imageName + ".jpg")
    
    if imgReturn is None:
        print("Error: Image is empty")
    else:   
        return imgReturn
        
def displayImage(inputImage: cv.Mat, caption: str = "01"): #par1 inputImage, par2 caption headline
    while(True):
        
        cv.imshow("Cap", inputImage)
        if cv.waitKey(15) == ord('f'):
            inputImage = cv.resize(inputImage, (screen_width, screen_height), interpolation=cv.INTER_CUBIC)

        if cv.waitKey(15) == ord('q'):
            break

def resize(inputImage: cv.Mat, scaleFactor: int = 1, fullScreenFlag: bool = 0):  #resize image
    if fullScreenFlag == 0:
        return cv.resize(inputImage, (inputImage.shape[1]*scaleFactor, inputImage.shape[0]*scaleFactor), interpolation=cv.INTER_CUBIC)
    return cv.resize(inputImage, (screen_width, screen_height), interpolation=cv.INTER_CUBIC)

def rotate(inputImage: cv.Mat, amount: int = 0):    #rotate image at user defined input 0-4

    if amount == 0:
        print("The image has been rotated precisely 0 degress")
        return inputImage
    
    for x in range(0, amount):
        rotatedImg = cv.rotate(inputImage, cv.ROTATE_90_CLOCKWISE)
    print("The image has been rotated " + "" + " times")
    return rotatedImg

def saveImage(inputImage: cv.Mat, fileName: str = "q"):
        
    if fileName == "q": #if the filename is q set filename to current timestamp
        now: datetime = datetime.now()
        Timestamp: str = now.strftime("%H_%M_%S")
        fileName: str = "Images/" + Timestamp + ".jpg"
    else:
        fileName: str = "Images/" + fileName + ".jpg"
        
    if cv.imwrite(fileName, inputImage) == True:
        print("Image saved to system")
    else:
        print("Error: could not save image")

def gaussianBlur(inputImage: cv.Mat, displayFlag: bool = False):
    kernel = np.ones((5,5),np.float32)/25
    bluredImage = cv.filter2D(inputImage,-1,kernel)
    
    if displayFlag == True:
        plt.subplot(121),plt.imshow(inputImage),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(bluredImage),plt.title('Gaussian blur output')
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    return bluredImage
        
def RGB2HVS(inputImage: cv.Mat):
    return cv.cvtColor(inputImage, cv.COLOR_BGR2HSV)

def isolateRedPixels(inputImage: cv.Mat, HSV: cv.Mat):

    lower_color = np.array([110,50,50], dtype=np.uint8)
    upper_color = np.array([130,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(HSV, lower_color, upper_color)
    #cv2.imshow('mask',mask)  
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(inputImage,inputImage, mask= mask)
    #cv2.imshow("b", res)

    imgray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(imgray,127,255,0)
    contours, hierarchy = cv.findContours(imgray,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    

    #erode = cv2.erode(res,None,iterations = 3)
    #dilate = cv2.dilate(erode,None,iterations = 10)
    #contours,hierarchy = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        x,y,w,h = cv.boundingRect(cnt)
        print (w,h)
        if(w < 60 or h < 60):
            continue

        cx,cy = x+w/2, y+h/2

        if 20 < res.item(cy,cx,0) < 30:
             #cv2.rectangle(f,(x,y),(x+w,y+h),[0,255,255],2)
            print ("yellow :", x,y,w,h)
        elif 100 < res.item(cy,cx,0) < 120:
             cv.rectangle(inputImage,(x,y),(x+w,y+h),[255,0,0],2)
             print ("blue :", x,y,w,h)
             detected = True
        return inputImage

def isolateChannel(inputImage: cv.Mat, channel: int):   #isolate a userdefined channel in multible colorspaced, limited to three channels
    b, g, r = cv.split(inputImage)
    
    if channel == 1: 
        return b


    if channel == 2: 
        return g

    if channel == 3: 
        return r

#noiceType: bytes = 1, noiceAmount: bytes = 128

def emulatedNoice(inputImage: cv.Mat, noiceType: str = 'rnd',noiceAmount: int = 70): #inputImage: image loaded in. noiceType: 1=gaussian 2=salt&peper

    if(noiceType == 'gaussian'):
        return np.array(255*random_noise(inputImage, mode='gaussian',amount = float(noiceAmount/100)), dtype = 'uint8')
       
    
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(inputImage, mode='s&p',amount = float(noiceAmount/100))

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')

    # Display the noise image
    cv.imshow('blur',noise_img)
    cv.waitKey(0)



def plotnoise(inputImage: cv.Mat):
    img = inputImage
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)

    # noise overlaid over image
    noisy = np.clip((img + noise*0.2),0,1)
    noisy2 = np.clip((img + noise*0.4),0,1)

    # noise multiplied by image:
    # whites can go to black but blacks cannot go to white
    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)

    # noise multiplied by bottom and top half images,
    # whites stay white blacks black, noise is added to center
    img2 = img*2
    n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)


    # norm noise for viz only
    noise2 = (noise - noise.min())/(noise.max()-noise.min())
    plt.figure(figsize=(20,20))
    plt.imshow(np.vstack((np.hstack((img, noise2)),np.hstack((noisy, noisy2)),np.hstack((noisy2mul, noisy4mul)),np.hstack((n2, n4)))))
    plt.show()
    plt.hist(noise.ravel(), bins=100)
    plt.show()
    
    
def cannyFilter(inputImage: cv.Mat):
    if(len(inputImage.shape)<3):
        
        edges = cv.Canny(inputImage, 100, 200)
            
    elif len(inputImage.shape)==3:
        
        print("Color(RGB)")
    
        edges = cv.Canny(cv.cvtColor(inputImage, cv.COLOR_BGR2GRAY),150,220)
        
    return edges

def houghCircle(inputImage: cv.Mat):
    
    circles	= cv.HoughCircles(inputImage, cv.HOUGH_GRADIENT, 1, minDist=100, param1=60, param2=40, minRadius=40, maxRadius=90)
    
    circles = np.uint16(np.around(circles))
    # Draw the circles
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(inputImage,(i[0],i[1]),i[2],(255,255,255),1)
        # draw the center of the circle
        cv.circle(inputImage,(i[0],i[1]),2,(255,0,0),3)
    cv.imshow('detected circles',inputImage)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img = readImage("Lag")   
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #houg = houghCircle(img)
    #gaussianBlur(img, True)
    #hsv = RGB2HVS(gaussianBlur(img, False))
    #print(houg)
    #edges =cannyFilter(img)
    #displayImage(edges)
    #plotnoise(img)
    emulatedNoice(img, 'gaussian')
      


if __name__ == "__main__":
    main()