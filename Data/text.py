import cv2
import numpy as np

img = cv2.imread('digits.png',0)
i=0
j=0
filename = "/mnisttxt/Number"
count=1
number_label=0
while(True):
	test = img[j:j+20, i:i+20]
	# cv2.imshow("Test", test)
	i += 20
	if i==2000:
		i=0
		j+=20
	if j>1000:
		break
	target = open(filename+str(number_label)+"_"+str(count)+".txt", 'w')
	test = test.flatten()
	line = 0
	for k in test:
		if line==399:
			target.write(str(k))
		else:
			target.write(str(k)+"\n")
		line += 1
	target.close()
	count+=1
	if count==501:
		number_label+=1
		count=1