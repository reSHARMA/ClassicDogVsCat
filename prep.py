from main import setRange,setState,getImg,shuffle
import scipy.misc as cv
import pandas as pd
import skimage.feature
from skimage.color import rgb2gray
datax = []
datay = []

def getImg(a,b):
    # requires a empty datax and datay array
    # 1 for dog
    # 0 for cat
    if a == 0 and b == 0:
        print("set a valid range using setRange()")
    else:
        for x in range(a,b):
            img = cv.imread("./train/dog." + str(x) + ".jpg")
            re_img = cv.imresize(img,(64,128))
            gimg = rgb2gray(re_img)
            hr = skimage.feature.hog(gimg, orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2))
            print(hr)
            datax.append(list(hr))
            datay.append(1)
            print("yes")
        for x in range(a,b):
            img = cv.imread("./train/cat." + str(x) + ".jpg")
            re_img = cv.imresize(img,(64,128))
            gimg = rgb2gray(re_img)
            hr = skimage.feature.hog(gimg, orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2))
            datax.append(list(hr))
            datay.append(0)
            print("no")

a,b = input().split()
a = int(a)
b = int(b)
getImg(a,b)
datax = list(datax)
datay = list(datay)
df1 = pd.DataFrame(data = datax)
df2 = pd.DataFrame(data = datay)
df = pd.concat([df1 , df2],axis = 1)
df = shuffle(df)
print(df.shape)
df.to_csv('data.csv')
