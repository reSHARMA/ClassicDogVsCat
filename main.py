import scipy.misc as cv
import skimage.feature
import numpy as np
from skimage.color import rgb2gray

# ckeck for max and min mage size
MinMaxSet = False
RangeSet = False
_Minx = 1080
_Miny = 1080
_a = 0
_b = 0
def getSetState():
    return MinMaxSet

def setState(x,y):
    _Minx , _Miny = x , y
    MinMaxSet = True

def ResetMinMax():
    MinMaxSet = False

def FindMinMax(a = _a,b = _b):
    if _a == 0 and _b == 0:
        _a , _b = a , b
        RangeSet = True
    if setState == False:
        for x in range(a,b):
            img = cv.imread("./ClassicCatvsDog/train/dog." + str(x) + ".jpg" )
            _x , _y , _z = img.shape
            _Minx = min(_x,_Minx)
            _Miny = min(_y,_Miny)
            print("new min shape" + _Minx + " " + _Miny)

def setRange(a,b):
    _a , _b = a ,b
    RangeSet = True

def getImg(a,b):
    # requires a empty datax and datay array
    # 1 for dog
    # 0 for cat
    _a , _b = a , b
    if a == 0 and b == 0:
        print("set a valid range using setRange()")
    else:
        for x in range(_a,_b):
            img = cv.imread("./train/dog." + str(x) + ".jpg")
            re_img = cv.imresize(img,(_Minx,_Miny))
            gimg = rgb2gray(re_img)
            hr = skimage.feature.hog(gimg, orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2))
            print("yes")
        for x in range(_a,_b):
            img = cv.imread("./train/cat." + str(x) + ".jpg")
            re_img = cv.imresize(img,(_Minx,_Miny))
            gimg = rgb2gray(re_img)
            hr = skimage.feature.hog(gimg, orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2))
            print("no")


def shuffle(df, n=1, axis=0):     
	df = df.copy()
	for _ in range(n):
		df.apply(np.random.shuffle, axis=axis)
	return df
