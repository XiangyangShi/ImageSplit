import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
import shutil
import sys, getopt

# paratmeters u need to provide
home=os.getcwd()+"/"
## this determined the mode this model works.
### 1 means detected the image provided in $singlepicturetruepaht
### 0 means run on all images and output on the folder of $writepicturepath
ifsinglepicture=1
singlepictruepath=home+'Test/94.jpg'
writepicturepath=home+'TestOutputNov22/'
## this determined whether you save the intermediate connecting componet
### i means to save all the component into $writefolder
ifwrite=0
writefolder=home+'/ppp/'
## ifcontour determine whether you draw a contour of the detected connectign component
## if allcontour determine whether you draw contours for all the componetents
ifcontour=0
ifallcontour=0
## take notes means whether you write a log
iftakenotes=0

# sys parameter noneed to change
SMsimilarity_treshold,BGsimilarity_treshold,HGsimilarity_treshold = 0.12,0.13,0.14
SMminarea,SMmaxarea = 70,1000
BGminarea,BGmaxarea = 80,2000
HGminarea,HGmaxarea = 300,8000
SMmarginbox,BGmarginbox,HGmarginbox = 3,10,15
BIGSIZEIMG,HUGESIZEIMG = 800 * 600 , 2 * 1024 * 798
imgB = []
thres = 0
margin_treshold = 250
labelCap = "0OABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelCap2 = [ "0","O","aa","bb","cc","dd","ee","ff","gg","hh","ii","jj" ]
labelCap3 = ["0,","O","Anew","Bnew","Cnew","Dnew","Enew"]
num = 5+2
labelCap = labelCap[0:num]
labelCap2 = labelCap[0:num]
labelCap3 = labelCap[0:num]

def usage():
	print "Example: Nov20.py -i 1 -f ~/a.jpg"
	print "-i 1 refers to singlepicture mode; must provide -f 'filename'"
	print "-i 0 refers to analyse all images in folder of ./ImageSplit/Test"
	print "-h for help"

def writeprint(path,strinput=home):
	print str(strinput)
	if iftakenotes:
		with open(path+'note.txt','a')as f:
			f.write(str(strinput))
			f.write('\n')

def wait(img):
	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
		cv2.imwrite('new.jpg',image)
		cv2.destroyAllWindows()

def cutmargin(inputimg,margin_treshold1):
	minY = 0
	minX = 0
	maxY = inputimg.shape[0]
	maxX = inputimg.shape[1]
	# print minX,maxX,minY,maxY
	for i in range(maxY):
		if(np.mean(inputimg[i,:])<margin_treshold1):
			minY=i
			break
	for i in range(maxY)[::-1]:
		if(np.mean(inputimg[i,:])<margin_treshold1):
			maxY=i
			break
	for i in range(maxX):
		if(np.mean(inputimg[:,i])<margin_treshold1):
			minX=i
			break
	for i in range(maxX)[::-1]:
		if(np.mean(inputimg[:,i])<margin_treshold1):
			maxX=i
			break
	# print minX,maxX,minY,maxY
	return inputimg[minY:maxY,minX:maxX]

def rightbottomBounding(inputimg,margin_treshold1):
	minY = 0
	minX = 0
	maxY = inputimg.shape[0]
	maxX = inputimg.shape[1]
	# print minX,maxX,minY,maxY
	for i in range(maxY)[::-1]:
		if(np.sum(margin_treshold1>np.mean(inputimg[i,:],axis=1))>10):
			maxY=i
			break
	for i in range(maxX)[::-1]:
		if(np.sum(margin_treshold1>np.mean(inputimg[:,i],axis=1))>10):
			maxX=i
			break
	# print minX,maxX,minY,maxY
	return maxX,maxY

def similarity(subimage,imga,imgname):
	maxsizeY = imga.shape[0]
	maxsizeX = imga.shape[1]
	score = 0
	res=cv2.resize(subimage,(maxsizeX,maxsizeY))
	for y in range(maxsizeY):
		for x in range(maxsizeX):
			imga_avg=np.mean(imga[y,x])
			res_avg=np.mean(res[y,x])
			mins=imga_avg-res_avg if imga_avg>res_avg else res_avg-imga_avg
			score = score + mins*mins/255.0/255.0
	score= score/maxsizeY/maxsizeX
	return score

# input [img to detected] ; output [[locations of labels]]
def findCharacter(img,outputs=False,outstr=home):
	result = []
	simtable = [0.3] * num
	mostpossiblelabel = []
	contourImg = []
	YY = img.shape[0]
	XX = img.shape[1]
	# noise removal
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
	writeprint(outstr,str(len(contours))+"contours in all")
	# 2 different sys to aynalze img in different size
	if XX * YY < BIGSIZEIMG :
		marginbox=SMmarginbox
		minarea=SMminarea
		maxarea=SMmaxarea
		writeprint(outstr,"smallsizeImg")
	elif XX * YY < HUGESIZEIMG:
		marginbox=BGmarginbox
		minarea=BGminarea
		maxarea=BGmaxarea
		writeprint(outstr,"bigsizeImg")
	else:
		marginbox=HGmarginbox
		minarea=HGminarea
		maxarea=HGmaxarea
		writeprint(outstr,"hugesizeImg")
	# filter all contours which area is not toobig or toosmall
	for contour in contours:
		x,y,w,h=cv2.boundingRect(contour)
		left = 0 if (x<=marginbox) else (x-marginbox)
		top = 0 if (y<=marginbox) else (y-marginbox)
		right = img.shape[1] if (x+w>=img.shape[1]-marginbox) else (x+w+marginbox)
		bottom = img.shape[0] if (y+h>=img.shape[0]-marginbox) else (y+h+marginbox)	
		area = cv2.contourArea(contour)
		if area>minarea and area<maxarea:
			ct=[contour]
			subimage=img[top:bottom,left:right]
			subimage=cutmargin(subimage,margin_treshold)
			contourImg.append( (subimage,ct,x,y,w,h) )
	# for everycontour mark them with 
	for subimage,ct,x,y,w,h in contourImg:
		maxsimhere=0.3
		indexoflabel=-1
		for i in range(num):
			imgA = imgB[i]
			sim = similarity(subimage,imgA,labelCap[i])
			if sim<maxsimhere:
				maxsimhere=sim
				indexoflabel=i
				#compare all template, find out the label with greatest likehood
		if indexoflabel!=-1:
			if maxsimhere<simtable[indexoflabel]:
				simtable[indexoflabel]=maxsimhere
				#update the greatest maxsim for each template
		mostpossiblelabel.append([indexoflabel,maxsimhere])
	print mostpossiblelabel
	writeprint(outstr,simtable)
	#since we have max sim
	counts=0;
	for j,sim in mostpossiblelabel:
		if sim<simtable[j]*1.1 and j>1:
			if  (XX*YY< BIGSIZEIMG and simtable[j]<SMsimilarity_treshold) or (XX*YY >= BIGSIZEIMG and XX*YY < HUGESIZEIMG and simtable[j]<BGsimilarity_treshold) or (XX*YY >= HUGESIZEIMG and simtable[j]<HGsimilarity_treshold):
				if ifcontour or ifallcontour:
					cv2.drawContours(img,contourImg[counts][1],-1,(0,255,0),3)
				result.append([contourImg[counts][2],contourImg[counts][3],contourImg[counts][2]+contourImg[counts][4],contourImg[counts][3]+contourImg[counts][5],labelCap[j],""])
				print "The difference between groundtruth & "+labelCap[j]+"score is :"+str(sim)
		else:
			if ifallcontour:
				cv2.drawContours(img,coutourImg[1],-1,(0,255,0),3)
		if ifwrite:
			cv2.imwrite(writefolder+labelCap[j]+str(sim)+'.jpg',contourImg[counts][0])
		counts=counts+1
	if outputs:
		cv2.imwrite(outstr+'result.jpg',img)
	else:
		cv2.imshow("result",img)
	# print x,y,w,h,len(contours)
	result.sort(key=lambda x:(x[4]),reverse=True)
	return result

#main func it can anaylse single picture at a time, false means whether write the result or not
def main(img,output=False,outstr=home):
	global imgB
	for filename in labelCap:
		imga=cv2.imread(home+'groundtruth/'+filename+'.jpg')
		imga=cutmargin(imga,margin_treshold)
		imgB.append(imga)
	res = findCharacter(img,output,outstr)
	print res
	# if len(res) == len( [i[5] for i in res if (i[4] == 'C')] ):
	# 	imgB=[]
	# 	print "2nd sys finding lowercases"
	# 	for filename in labelCap2:
	# 		imga=cv2.imread('/Users/shixiangyang/Desktop/groundtruth/'+filename+'.jpg')
	# 		imga=cutmargin(imga,margin_treshold)
	# 		imgB.append(imga)
	# 	res = findCharacter(img,output,outstr)
	# 	print res
	# if res==[]:
	# 	imgB=[]
	# 	print "3rd sys finding Times New Roman"
	# 	for filename in labelCap3:
	# 		imga=cv2.imread('/Users/shixiangyang/Desktop/groundtruth/'+filename+'.jpg')
	# 		imga=cutmargin(imga,margin_treshold)
	# 		imgB.append(imga)
	# 	res = findCharacter(img,output,outstr)
	# 	print res
	writeprint (outstr,res)
	count = 0
	#b and b prime will be detected
	#make them becomes b1 b2 b3 b4
	for index in range(len(res)-1):
		if res[index][4]==res[index+1][4]:
			count=count+1
			res[index][5]=str(count)
			res[index+1][5]=str(count+1)
		else:
			count=0
	for k in res:
		imgK=img[k[1]:,k[0]:]
		maxwidth,maxheight=rightbottomBounding(imgK,margin_treshold)
		if output:
			cv2.imwrite(outstr+k[4]+k[5]+'.jpg',imgK[:maxheight,:maxwidth])
		else:
			cv2.imshow(k[4]+k[5],imgK[:maxheight,:maxwidth])
		img[k[1]:,k[0]:]=255*np.ones((imgK.shape[0],imgK.shape[1],3))

# call main , findout all the picutre and run the function
def main2():
	for i in range(1,2400):
		imgToDetected= cv2.imread(home+'/Test/'+str(i)+'.jpg')
		newpath=writepicturepath+str(i)+'/'
		if not os.path.exists(newpath):
			os.mkdir(newpath)
		main(imgToDetected,True,newpath)
		print "the No. "+str(i)+" fininshed"

opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
for op, value in opts:
	if op == "-i":
		ifsinglepicture=eval(value)
	elif op == "-f":
		singlepictruepath=value
	elif op == "-h":
		usage()
		sys.exit()
# if __name__=='__main__':
if ifsinglepicture:
	img = cv2.imread(singlepictruepath)
	main(img)
	wait(img)
else:
	main2()
