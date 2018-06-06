# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:55:17 2018

@author: JVM
"""

"""
This demonstrates the use of the library of modules developed. The aim is to use input image of chlorophyll autofluorescence
and fluorescent protein labelled nuclei to detect gemmae outlines and nuclei positions. This information is standardised
for shape to a unit circle and a 2D histogram of nuclei density is created.

For more information see Quantifying Promoter Element Activation in Marchantia polymorpha Gemmae by J V Mante
"""

import matplotlib.pyplot as plt

filedirect = "C:\\Images\\"
notchimname = filedirect + "007.tif"
chll_im_name = filedirect + "008.tif"
ubiqimname = filedirect + "009.tif"


#find outline
cntx,cnty,cnt,area, perimeter, center, radius = contourpoints (chll_im_name)

#find MpARF3:Venus (ubiq) and MpYuC2:mTurquoise (notch) nuclei
unucx,unucy, unucradii, unucbright,unumofnuc, unucoutside, unumdelpts = findnuclei(ubiqimname, cnt = cnt, mask = 1)
nnucx,unucy, nnucradii, nnucbright,nnumofnuc, nnucoutside, nnumdelpts = findnuclei(notchimname, cnt = cnt, mask = 1)


#Display found nuclei over image
nucleioverimage(ubiqimname, unucx, unucy, unucradii, unucbright, savefig = False)
nucleioverimage(notchimname, nnucx, nnucy, nnucradii, nnucbright, savefig = False)

#find notches using ubiq nuclei
nx,ny,nx1,ny1,nx2,ny2 = findnotches (chll_im_name, cnt, cntx, cnty, unucx, unucy)

#plot ubiq
plt.figure()
plt.scatter(unucx,unucy, color="blue")
plt.scatter(cntx,cnty, color = "orange")
plt.scatter(nx,ny, color = "black")

#plot notch
plt.figure()
plt.scatter(nnucx,nnucy, color="blue")
plt.scatter(cntx,cnty, color = "orange")
plt.scatter(nx,ny, color = "black")

#Translation and Rotation
utn1x, utn1y, utn2x, utn2y, utnucx,utnucy,tcntx,tcnty = transrotateshape (nx1, ny1, nx2, ny2, unucx, unucy, cntx, cnty)
ntn1x, ntn1y, ntn2x, ntn2y, ntnucx,ntnucy,tcntx,tcnty = transrotateshape (nx1, ny1, nx2, ny2, nnucx, nnucy, cntx, cnty)


#Circularise
ucnucx,ucnucy,usnucradii, usnucbright = circularise(utnucx, utnucy, tcntx, tcnty, unucradii, unucbright)
ncnucx,ncnucy,nsnucradii, nsnucbright = circularise(ntnucx, ntnucy, tcntx, tcnty, nnucradii, nnucbright)

#Show 2D histograms
outputplots(ucnucx,ucnucy,usnucradii, usnucbright, verbosity = False,heatblockplt = True )
outputplots(ncnucx,ncnucy,nsnucradii, nsnucbright, verbosity = False,heatblockplt = True )