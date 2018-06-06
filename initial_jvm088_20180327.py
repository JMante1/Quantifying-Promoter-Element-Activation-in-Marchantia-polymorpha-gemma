# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:48:57 2017

@author: JVM, jvm34@cam.ac.uk

"""

################################################################################
def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, 1 )
        Input data
    s : scalar or array_like, shape (n, 1) 
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Example
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection

###############################################################################
def pol2cart(rho, phi):
    
    """
    This function converts polar coordinates (distance and angle) to cartesian (x and y) coordinates
    
    Polar:
        rho = the distance from (0,0)
        phi = angle, in RADIANS, from the positive x-axis.
    
    Parameters
    -----------
    The function can accept either a single value (scalar) for each parameter, or multiple inputs (array_like)
    Rho, phi: scalar or array_like, shape (n,1)
    
    Returns
    -------
    x,y: scalar or array_like, shape(n,1)
    
    Example
    --------
    x,y = pol2cart(1,np.pi)
    print(x,y)
    """
    import numpy as np
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
################################################################################
def cart2pol(x, y):
    
    """
    This function converts cartesian co-ordinates (x and y) to polar coordinates (distance and angle)
    
    Polar:
        rho = the distance from (0,0)
        phi = angle, in RADIANS, from the positive x-axis.
    
    Parameters
    -----------
    The function can accept either a single value (scalar) for each parameter, or multiple inputs (array_like)
    
    x, y: scalar or array_like, shape (n,1)
    
    Returns
    -------
    rho,phi: scalar or array_like, shape(n,1)
    
    Example
    --------
    rho, phi = cart2phol(1,1)
    print(x,y)
    """
    import numpy as np
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

######################################################################################
def contourpoints (chll_im_name, minthresh=1,noisy=True, verbosity=False, contdefect=False):
    """    
    This function tries to find the edge of the object to separate it from the background.
    It finds the largest outline in an image and returns the x and y values of 900 points describing this outline
    
    Parameters
    ----------
    chll_im_name: string
        Provides the name of the image to find the contours on, including the 
        full file path
    minthresh: scalar, optional, default: 1
        The cutoff value for binarisation. Any values above this will be
        converted to white once the image is greyscaled
    noisy: Boolean, optional, default: True
        If the image is noisy input True otherwise leave as False. A value of True leads to blurring
        being used to remove noise from the image
    verbosity: Boolean, optional,  default: False
        If true an image with the found contours is displayed
    contdefect: Boolean, optional, default: False
        If true then the largest contour defect is corrected regardless of the size of the defect
        
    Returns
    --------
    cntx,cnty: aray_like, shape (n,1)
        Cartesian co-odinates of the contour points with the bottom left corner being (0,0)
    cnt: array_like, shape (n,1,2)
        each row has double square brackets
    area: integer
        The area, in pixels, that the outline encloses
    perimeter: scalar
        The perimeter, in pixels of the outline
    center: tuple
        The center of the minimum enclosing circle for the outline
    radius: integer
        The length of the radius of the minimum enclosing circle for the outline
        
    Example
    --------
    cntx,cnty,cnt,area, perimeter, center, radius = contourpoints('C:/Python34/images/image.tif',15)
    plt.scatter(cntx,cnty)
    """
    
    #imports
    import cv2
    import numpy as np
    import math
    from pyefd import elliptic_fourier_descriptors

    
    #Read image
    im = cv2.imread(chll_im_name)
      
    #convert image to grayscale
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    #image size
    height, width, channels = im.shape
    
    #blur
    """
    if the image is noisy blur to remove noise
    """
    if noisy:
        imgray = cv2.blur(imgray,(5,5))
        
    #binarise image (Each grayscale pixel is converted either to or o or 255 (black or white))
    """
    255 means that any pixels avove minthresh will be given the color 255 i.e. white
    
    minthres: minthres and higher grayscale values are mapped to 255, others to 0
    Thresh: output threshold image
    ret   : threshold value found when using things like otsu binarisation
    """
    ret, thresh = cv2.threshold(imgray, minthresh, 255, cv2.THRESH_BINARY)
    
    #Find Contours
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    

    
    """
    Once outlines are found, find the one with most points among them (assumption: this will be the longest one) 
    """
    #Number of Contours found
    lencnt = len(contours)
    
    #Initialise array 1 column lencnt rows
    cnt = np.ones((lencnt,1)) 
    
    #Create list of number of points in each of the contours found
    for i in range(0,lencnt):
        cnt[i] = len(contours[i])
        
    #Find contour with the greatest number of points
    maxcntind = np.argmax(cnt)
    cnt = contours[maxcntind]
    
       
    """Find hull (completely convex shape surrounding the contour)"""
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    """Find Convexity defects"""
    defects = cv2.convexityDefects(cnt,hull)
    
    """Sort the Convexity defects by distance from hull"""
    dist = np.ones((len(defects),1))
    start = np.ones((len(defects),2))
    end = np.ones((len(defects),2))
    far = np.ones((len(defects),2))
    
    for i in range(defects.shape[0]):
        #s is start point, e is end point, f is fartherst point, d is distances between contour and hull
        s,e,f,d = defects[i,0]
        dist[i] = d
        start[i] = tuple(cnt[s][0])
        end[i]= tuple(cnt[e][0])
        far[i] = tuple(cnt[f][0])
    
    distsort = np.argsort(dist, axis = 0) 
    maxdistind = distsort[len(distsort)-1][0]
    maxdist = dist[maxdistind][0]
    
    """Minimum enclosing circle to give a distance conversion"""
    #minimum enclosing circle
    (xc,yc),radius = cv2.minEnclosingCircle(cnt)
    center = (int(xc),int(yc))
    radius = int(radius)
    
    """
    Remove largest contour defect if it is large enough, or user says to.
    The defect is replaced by a slight indent to prevent loss of notch
    """
    xin=0
    yin=0
    if (maxdist/radius) >50 or contdefect:
        #create matrix with details about the largest contourdefects
        st = defects[maxdistind][0][0]
        en = defects[maxdistind][0][1]
        fr = defects[maxdistind][0][2]
        farpt =  tuple(cnt[fr][0])
        startpt = tuple(cnt[st][0])
        endpt = tuple(cnt[en][0])
        
        
        #Set up points to work with mathematically
        startpty = startpt[1]
        endpty = endpt[1]
        startptx = startpt[0]
        endptx = endpt[0]
        farptx = farpt[0]
        farpty = farpt[1]
        
        #Distance startpt to end point
        sttoenddist = np.sqrt((np.square(startptx-endptx)+np.square(startpty-endpty)))
        
        #Distance inward from convex hull required
        distanceinfornotch = sttoenddist*np.tan(math.pi/10)
        
        #Find midpoint between start and end point of contour defect
        midptx = (startptx+endptx)/2
        midpty = (startpty+endpty)/2
        
        #Find equation of line conecting the start and end point of the contour defect,
        #1/inmline is the gradient of the line
        if endpty == startpty:
            y1 = endpty
            y2 = endpty
            x1 = midptx + distanceinfornotch
            x2 = midptx - distanceinfornotch
        elif endptx == startptx:
            x1 = endptx
            x2 = endptx
            y1 = midpty + distanceinfornotch
            y2 = midpty - distanceinfornotch
        else:
            inmline = -1*((startptx-endptx)/(startpty-endpty))
            c = midpty-(inmline)*midptx
            
        
            #Find points on line perpendicular to start to end line going through midpt a 
            #distance distanceinfornotch away from the midpoint
            bquad = -2*(inmline*midpty - inmline*c + midptx)
            aquad = 1 + np.square(inmline)
            cquad = -np.square(distanceinfornotch) + np.square(midpty) - 2*midpty*c + np.square(c) + np.square(midptx)
            
            
            x1 = (-bquad + np.sqrt(np.square(bquad)-4*aquad*cquad))/(2*(aquad))
            x2 = (-bquad - np.sqrt(np.square(bquad)-4*aquad*cquad))/(2*(aquad))
            
            y1 = inmline*x1 + c
            y2 = inmline*x2 + c
        
        #Find distances of the two points to the fartherest inward point of the contour defect
        fdist1 = np.square(farptx-x1)+np.square(farpty-y1)
        fdist2 = np.square(farptx-x2)+np.square(farpty-y2)
        
        
        #Select the point that is further in to be the midpoint of the inward dent
        if fdist1>fdist2:
            xin = x2
            yin = y2
        else:
            xin = x1
            yin = y1      
        notchinpt = tuple((int(xin),int(yin)))
        
        #Draw two white lines on the image from start to inward point and inward point to endpoint
        imwithline = cv2.line(im,startpt,notchinpt,[255,255,255],2)
        imwithline = cv2.line(imwithline,notchinpt,endpt,[255,255,255],2)
        
        """
        Repeat outline finding with the image with lines on it
        The outline will now follow the lines rather than going further in
        """
        #convert image to grayscale
        imgrayln = cv2.cvtColor(imwithline, cv2.COLOR_BGR2GRAY)
        
        #image size
        height, width, channels = imwithline.shape
        
        #blur
        """
        if the image is noisy blur to remove noise
        """
        if noisy:
            imgrayln = cv2.blur(imgrayln,(5,5))
            
        #binarise image (Each grayscale pixel is converted either to or o or 255 (black or white))
        """
        255 means that any pixels avove minthresh will be given the color 255 i.e. white
        
        minthres: minthres and higher grayscale values are mapped to 255, others to 0
        Thresh: output threshold image
        ret   : threshold value found when using things like otsu binarisation
        """
        ret, thresh = cv2.threshold(imgrayln, minthresh, 255, cv2.THRESH_BINARY)
        
        #Find Contours
        im2ln, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        """
        Once contours are found, find the one with most points among them (assumption: this will be the longest one) 
        """
        #Number of Contours found
        lencnt = len(contours)
        
        #Initialise array 1 column lencnt rows
        cnt = np.ones((lencnt,1)) 
        
        #Create list of number of points in each of the contours found
        for i in range(0,lencnt):
            cnt[i] = len(contours[i])
            
        #Find contour with the greatest number of points
        maxcntind = np.argmax(cnt)
        cnt = contours[maxcntind]
        
    """
    Prepare the output arrays by reading the values of the contour with the most points 
    """
    #create x and y vertical arrays
    xcnt = np.ones((len(cnt),1))
    ycnt = np.ones((len(cnt),1))
    
    #Add x and y values to the arrays
    for i in range(0,len(cnt)):
        xcnt[i] = cnt[i][0][0]
        ycnt[i] = cnt[i][0][1]
    
    
    
    """Elliptical Fourier- smoothing"""
    edgepnts = np.append(xcnt, ycnt,axis = (1))
    coeffs = elliptic_fourier_descriptors(edgepnts, order=20, normalize=False)
    
    #Setting values
    locus = (np.average(xcnt),np.average(ycnt))
    numpoints = 9000
    
    t = np.linspace(0,1.0, numpoints)

    xt = np.ones((numpoints,)) * locus[0]
    yt = np.ones((numpoints,)) * locus[1]
    
    #Mapping x and y for thetas
    for n in range(0,coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t))
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t))
    
    
    cntx = xt
    cnty = yt
    
    
    #Create OpenCV readable contour from smoothed points
    L = []
    
    for i in range(0, numpoints):
        xpoint = cntx[i]
        ypoint = cnty[i]
        points = np.append(xpoint,ypoint)
        L = np.append(L,points)
    cnt = np.array(L).reshape((-1,1,2)).astype(np.int32)    
    
    """Output Image"""
    if verbosity:
        #Reload im as want output without white lines on it
        im = cv2.imread(chll_im_name)
        
        #Draw largest contour on the image
        cv2.drawContours(im, [cnt], -1, (0,255,0), 1)
        
        #Create new image name
        pointind = chll_im_name.find('.')
        pointindmin3 = pointind-3
        seriesname = chll_im_name[pointindmin3:pointind]
        nucpointimname = chll_im_name[:pointindmin3]
        nucpointimname = nucpointimname+"contour_"+seriesname + ".tif"
       
        #Save file
        cv2.imwrite(nucpointimname,im)
        
    #Correct for image co-ordinates versus cartesian
    for i in range(0,len(cnty)):
        cnty[i] = height - cnty[i]   
    #Calculate Area and Perimeter
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    
    return (cntx,cnty,cnt,area, perimeter, center, radius)

#########################################################################################################
def pointincontour( x, y,vertices=None, Polygon=True,radius =None, center = None):
    """
    Deterines whether a given point is within a given polygon/circle
    
    Parameters
    ---------
    vertices: array_like, shape (n,2), optional, default:None
        Required if Polygon is True
        Vertices of the polygon to test if the point lies within the polygon
    x,y: scalars
        Cartesian co-odinates of the point to test
    Polygon: Boolean, optional, default:True
        If true then checks if the point is within a polygon
        If false then checks if the point is within a circle
    radius: scalar float, optional, default:None
        If Polygon is false then a radius is required for the circle
    center: tuple, optional, default:None
        If Polygon is false than the center of the circle is required
    
    Returns
    -------
    
    True or False
        True if the point is within the polygon/circle, false otherwise
    
    Example1
    -------
    vertices = [(1 , 1),(-1,-1),(-1,1), (1,-1)]
    pointincontour(0,0, vertices) 
    
    Example2
    -------
    pointincontour(1, 0,Polygon=False, radius = 5, center = (0,0))
    
    """
    from matplotlib.path import Path as mpPath
    import numpy as np
    
    if Polygon:
        path =mpPath(vertices)
        point = (x,y)
        Answer = path.contains_point(point)
    else:
        if np.square((x-center[0]))+np.square((y-center[1]))<np.square(radius):
            Answer = True
        else:
            Answer = False
    return Answer

#######################################################################################################################
def findnuclei(imagename, cnt = None, mask = 1, radius = None, center = None, verbosity = False, \
               minthreshold = 0, maxthreshold = 255, threshstep = 1 , minblobdist=0, \
               filterbycolor=True, blobcol=255, filterbyarea=True, minarea=0, maxarea=2, \
               filterbycircul=True, mincircul = 0 , maxcircul= 1, filterbyconvex=True, \
               minconvex = 0, maxconvex= 1, filterbyinertia=True, mininert=0,\
               maxinert=1):

    """
    This function finds blobs of interest, e.g. nuclei and seperates them from
    the background. 
    It returns the x and y values of the centrres of the points as well as
    size and brightness and the total number of points found.
    
    Explanation of blob detection adapted from: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    
    Then Function has several steps:
        -load image
        -turrn image greyscale
        -detect blobs:
            -thresholding - converts source image to severral binary images by
            thresholding the source image. Thresholds start at minthreshold,
            incremented by threshstep until maxthreshold
            -grouping - in each binary image connected white pixels are grouped
            together as blobs
            -merging - the centers of the blobs in all binary images are calculated
            and blobs closer together than minblobdist arre merged
            -blobs are discarded if they do not meet criteria:
                -blob color: if used then can look for dark or light blobs
                -size: can set a minimum and maximum area
                -circularity: measures how close to a circle the shape is.
                    Circularity = 4*pi*Area/(perimeter)^2. So a circle
                    has cirrcularity 1 and a square .785 etc.
                -convexity: Area of blob/Area of convex hull. Complex hull is 
                    tightest convex shape that completely encloses shape.
                -inertia: Measures linearity of a shape. 1 for a circle and 0
                    for a line. For an ellipse somewhere inbetween. It is calculated
                    as the sum of r^2 where r is the distance of points from the
                    center of mass
            -center and radius calculation - centers and radii of new meged blobs
            are calculated and returned
        -output data
     
    
    Parameters
    -----------
    imagename: string
        Provides the name of the image to find the points on, including the 
        full file path
    verbosity: Boolean, optional, default:False
        If true than an image with nucle circled named test_imagename.tif will
        be saved
    minthreshold: scalar, optional, default:0
        The minimum greyscale value of blobs to be detected
    maxthreshold: scalar, optional, default:255
        The maximum minimum greyscale value of blobs to be detected
    threshstep: scalar, optional, default:1
        The number of differrent binary images for blobs to be detected at
    minblobdist: scalar, optional, default:0
        The minimum number of pixels between blobs
    filterbycolor: Boolean, optional, default:True
        If true blobs will be filtered by color
    blobcol: scalar 255 or 0, optional, default:255
        If 255 and filterby color is true then white blobs, if zero then black
        blobs
    filterbyarea:Boolean, optional, default:True
        If true blobs will be filtered by area
    minarea: scalar, optional, default:0
        The minimum area a blob must have
    maxarea: scalar, optional, default:2
        The maximum area a blob must have
    filterbycircul: Boolean, optional, default:True
        If true blobs will be filtered by circularity
    mincircul: scalar, optional, default:0
        The minimum circularity (no less than 0)
    maxcircul: scalar, optional, default:1
        The maximum circularity (no more than 1)
    filterbyconvex: Boolean, optional, default:True
        If true filter by convexity
    minconvex: scalar, optional, default:0
        The minimum convexity (0<= x<= 1)
    maxconvex: scalar, optional, default:1
        The maximum convexity (minconvex< x <=1)
    filterbyinertia: Boolean, optional, default:True
        If true filter by inertia
    mininert: scalar, optional, default:0
        The minimum inertia (0<= x <= 1)
    maxinert: scalar, optional, default:1
        The maximum inertia (mininert < x <= 1)
    cnt: array_like, shape (n,1,2), optional,default=None:
        Outline necessary if masking by outline
    mask: scalar 0,1, or 2, optional, default:1
        0 means no masking
        1 means no nuclei found outside supplied outline
        2 means no nuclei found outside the (minimum enclosing) circle
    radius: scalar, optional, default:None
        Required if mask is 2
        The radius of the circle to use as mask
    center: tuple, optional, default:None
        Required if mask is 2
        The center of the circle to use as mask
   
        
    Returns
    -------
    nucx,nucy = aray_like, shape (n,1)
        Cartesian co-odinates of the found points witht the bottom left corner of the photo being (0,0)
    nucradii = aray_like, shape (n,1)
        Radii of found points
    nucbright = aray_like, shape (n,1)
        Brightness of found points 
    numofnuc = scalar
        The number of points found
    numdelpts = scalar
        The number of points removed due to overlapping with other points
    nucoutside = scalar
        The number of points found outside the outline of the mask
        
    
    
    Example
    --------
    nucx,nucy, nucradii, nucbright,numofnuc, nucoutside, numdelpts = findnuclei('C:/Python34/images/image.tif',mask=0)
    plt.scatter(nucx,nucy)
    """
    """ Import necessary functions"""
    import cv2
    import numpy as np
    
    
    
    """Read Image"""
    img = cv2.imread(imagename, 1)  #1 so color, if zero than greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    """Set Blob Detection Parameters"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = minthreshold
    params.maxThreshold = maxthreshold
    params.thresholdStep = threshstep
    params.minDistBetweenBlobs = minblobdist
    params.filterByColor = filterbycolor
    params.blobColor = blobcol
    params.filterByArea = filterbyarea
    params.minArea = minarea
    params.maxArea = maxarea
    params.filterByCircularity = filterbycircul
    params.minCircularity = mincircul
    params.maxCircularity = maxcircul
    params.filterByConvexity = filterbyconvex
    params.minConvexity = minconvex
    params.maxConvexity = maxconvex
    params.filterByInertia = filterbyinertia
    params.minInertiaRatio = mininert
    params.maxInertiaRatio = maxinert
    
    """Detect Blobs"""
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    """Remove Keypoints that overlap"""
    keypt1 = keypoints
    numdelpts = 0
    
    for i in range (0, len(keypoints)):
        deletepoint = False
        for j in range (0, len(keypt1)):
            retval = cv2.KeyPoint_overlap(keypoints[i],keypt1[j])
            if retval == 1:
                iisj = j
            elif retval > 0 and retval < 1:
                deletepoint = True
        if deletepoint:
            keypt1 = np.delete(keypt1, (iisj), axis = 0)
            numdelpts = numdelpts+1

    keypoints = keypt1    
    
    """Configure Output Arrays"""
    #create x,y, radii, brightness vertical arrays
    nucx = [0]
    nucy = [0]
    nucradii = [0]
    nucbright = [0]
    nucout = []
    
    tupcnt = np.ones((len(cnt),2))
    for i in range(0,len(cnt)):
        tupcnt[i][0] =cnt[i][0][1]
        tupcnt[i][1] =cnt[i][0][0]
    
    """Apply Mask"""
    #Fill arrays
    i = 0

    for h in range(0, len(keypoints)):
        keyPoint = keypoints[h]
        if mask == 1:
            if pointincontour(keyPoint.pt[1], keyPoint.pt[0], vertices = tupcnt):
                nucx = np.vstack((nucx, keyPoint.pt[0]))
                nucy = np.vstack((nucy, (1024 - keyPoint.pt[1])))
                nucradii = np.vstack((nucradii, keyPoint.size))
                nucbright = np.vstack((nucbright, img[round(int(nucx[i+1])),round(int(nucy[i+1]))]))
                i = i + 1
            else:
                nucout = np.append(nucout, h)
        elif mask == 2:
            if pointincontour(keyPoint.pt[0], keyPoint.pt[1], Polygon = False, radius = radius, center = center):
                nucx = np.vstack((nucx, keyPoint.pt[0]))
                nucy = np.vstack((nucy, (1024 - keyPoint.pt[1])))
                nucradii = np.vstack((nucradii, keyPoint.size))
                nucbright = np.vstack((nucbright, img[round(int(nucx[i+1])),round(int(nucy[i+1]))]))
                i = i + 1
            else:
                nucout = np.append(nucout, h)

        else:
            nucx = np.vstack((nucx, keyPoint.pt[0]))
            nucy = np.vstack((nucy, (1024 - keyPoint.pt[1])))
            nucradii = np.vstack((nucradii, keyPoint.size))
            nucbright = np.vstack((nucbright, img[round(int(nucx[i+1])),round(int(nucy[i+1]))]))
            i = i + 1 
    
    nucoutside = len(nucout)          
    nucx = np.delete(nucx, (0), axis=0)
    nucy = np.delete(nucy, (0), axis=0)
    nucradii = np.delete(nucradii, (0), axis=0)
    nucbright = np.delete(nucbright, (0), axis=0)
    numofnuc = len(nucx)

    
    """Image Output"""   
    if verbosity:
        #Create new image name
        pointind = imagename.find('.')
        pointindmin3 = pointind-3
        seriesname = imagename[pointindmin3:pointind]
        nucpointimname = imagename[:pointindmin3]
        nucpointimname = nucpointimname+"CircledNuclei_"+seriesname + ".tif"
        
        # Draw detected blobs as red circles.
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if mask ==1:
            im_with_keypoints = cv2.drawContours(im_with_keypoints, [cnt], -1, (0,255,0), 2)
        elif mask ==2:
            im_with_keypoints = cv2.circle(im_with_keypoints,center,radius,(0,255,0),2)
        #Save file
        cv2.imwrite(nucpointimname,im_with_keypoints)
    
    
    return(nucx,nucy, nucradii, nucbright,numofnuc, nucoutside, numdelpts)





#################################################################################################
def findnotches (chllim, cnt, cntx, cnty, nucx, nucy, imagename = None, verbosity = False, \
                 userinput = False, numpossiblenotch= -1, numnotch = 2):
    """
    This finds the largest numpossiblenotch (e.g. 4) indents in the contour 
    and selects the two with the greatest nuclei density.
    It can work on its own or with human input.
    
    Required Functions
    ------------------
    pointinpoly
    
    Parameters
    ----------
    chllim: string
        Provides the name of the image to find the contour deffects on, 
        including the full file path
    cntx,cnty: aray_like, shape (n,1)
        Cartesian co-odinates of the contour points with the bottom left corner being (0,0)
    cnt: array_like, shape (n,1,2)
        list of all contour points for the contour the indents are to be found 
        on. in the format [[[x1, y1]],.. [[xn, yn]]]
    nucx, nucy: array_like, shape (n,1)
        x and y co-ordinates of nuclei
    verbosity: Boolean, optional, default: False
        If true and userinput is false then an image with the possible notches 
        and contour will be shown
    userinput: Boolean, optional, default: false
        If true then the user selects the best two notches from a plot showing 
        numpossiblenotch posibilities
    numpossiblenotch: scalar, default: -1
        The number of possible notches for which the nuclear density should be 
        investigated, either by the computer or by
        the user
        If negative the numpossible notches depends on whether or not user input is used
            It becomes 6 if userinput is True or 4 otherwise
    numnotch: scalar, optional, default:2
        The number of notches to find
    imagename: string, optional, default:None
        If verbosity is True
        Provides the name of the output image (including the full file path)
        
    Returns
    -------
    nx,ny: aray_like, shape (n,1)
        The x and y co-ordinates of all notches
    nx1, ny1: scalars
        The x and y co-ordinate of the right most notch, or if more than 2 notches the first notch found
    nx2, ny2: scalars
        The x and y co-ordinate of the leftt most notch, or if more than 2 notches the second notch found
        
    Example
    -------
    nx,ny, nx1, ny1, nx2, ny2 = findnotches('C:/Python34/images/image.tif',cnt, nucx, nucy)
    plt.scatter(cntx,cnty)
    plt.scatter(nucx,nucy)
    plt.scatter(nx,ny)
    
    """
    
    
    """Import necessary functions"""
    import cv2
    import numpy as np    
    import matplotlib.pyplot as plt
    from matplotlib import pylab
    
    """Set number of possible notches"""
    if numpossiblenotch < 0 and userinput:
        numpossiblenotch = 6
    elif numpossiblenotch<0 and not userinput:
        numpossiblenotch = 4
        
    
    """Find hull (completely convex shape surrounding the contour)"""
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    """Find Convexity defects"""
    defects = cv2.convexityDefects(cnt,hull)
    
    """Sort the Convexity defects by distance from hull"""
    defdist = [0]
    deffar = [0]
    
    for i in range(defects.shape[0]):
        #s is start point, e is end point, f is fartherst point, d is distances between contour and hull
        s,e,f,d = defects[i,0]
        defdist = np.vstack([defdist, d])
        deffar = np.vstack([deffar, f])
    defdist = np.delete(defdist, (0), axis = 0)
    deffar = np.delete(deffar, (0), axis = 0)
    
    
    sordefdist = np.argsort(defdist, axis = 0) #indices of what defdist would be when sorted by distance
    lensordefdist = len(sordefdist) # number of convexity defects
    
    """Find numpossiblenotch biggest convexity defects"""
    #Read image
    im = cv2.imread(chllim)
    
    #set up arrays
    inlarge = np.ones((numpossiblenotch,1))
    distdef = np.ones((numpossiblenotch,1))
    xd = np.ones((numpossiblenotch,1))
    yd = np.ones((numpossiblenotch,1))
    
    #fill arrays, including those necessary for creation of an image
    for i in range (0, numpossiblenotch):
        inlarge[i] = sordefdist[lensordefdist - (i+1)] # find index of greatest numpossiblenotch indices
        distdef[i] = defdist[int(inlarge[i])] # find distances associated with these indices
        s,e,f,d = defects[int(inlarge[i]),0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        xd[i] = far[0]
        yd[i] = 1024 - far[1]
        cv2.line(im,start,end,[0,255,0],2)
        cv2.circle(im,far,5,[0,0,255],-1)
    
    #Show image if verbosity is true and userinput is false
    if verbosity and not userinput:
        #Create new image name
        pointind = imagename.find('.')
        pointindmin3 = pointind-3
        seriesname = imagename[pointindmin3:pointind]
        nucpointimname = imagename[:pointindmin3]
        nucpointimname = nucpointimname+"NotchFinding_"+seriesname + ".tif"
        cv2.imwrite(nucpointimname,im)

    """Decide which indents to use, either by use or nuclei density"""
    if userinput:
        #Show plot with all possible notches
        plt.figure()
        pylab.figure(figsize=(10,10))
        plt.scatter(nucx,nucy)
        plt.scatter(cntx,cnty)
        plt.scatter(xd,yd, c = "black")
        for i in range(0,numpossiblenotch):
            plt.text(xd[i],yd[i]+10, i, fontsize = 20, weight = 1000, color="black",backgroundcolor="white")
        plt.show(block=False)
        
        #Ask user to decide which of the four points are notches
        numofnotch = int(input("Number of Notches = "))
        
        nxo = []
        nx = []
        ny = []
        
        for i in range (0, numofnotch):
            nxno1 = int(input(str(i+1)+" Notch Number = "))
            nxo = np.append(nxo,nxno1)
            nx =  np.append(nx,xd[nxno1])
            ny = np.append(ny, yd[nxno1])
    else:
        #finding nuclei density around each possible notch
        sqsize = 50 #area around the notch to use
        numpoint = np.zeros((numpossiblenotch,1)) #initialise aray
        
        #Check for a square around each possible notch how many nuclie there are
        for i in range(0,numpossiblenotch):
            posx = xd[i]
            posy = yd[i]
            vertices = [(int(posx+sqsize) , int(posy)),(int(posx-sqsize) , int(posy)),(int(posx),int(posy+sqsize)), (int(posx),int(posy-sqsize))]
            for point in range(0,len(nucx)-1):
                if pointincontour( nucx[point],nucy[point],vertices=vertices):
                    numpoint[i] = numpoint[i] + 1
        
        #Find the possible notches witht he greatest nuclei density
        sornumpoint = np.argsort(numpoint, axis = 0)
        lensornumpoint = len(sornumpoint)
        
        nx = []
        ny = []
        
        for i in range (0, numnotch):
            notchind = sornumpoint[lensornumpoint-(i+1)]
            nx =  np.append(nx,xd[notchind])
            ny = np.append(ny, yd[notchind])

    #Make n1 the right most notch
    if len(nx) ==2:
        nxa = nx[0]
        nya = ny[0]
        nxb = nx[1]
        nyb = ny[1]
        if nxa>nxb:
            nx1 = nxa
            ny1 = nya
            nx2 = nxb
            ny2 = nyb
        else:
            nx1 = nxb
            ny1 = nyb
            nx2 = nxa
            ny2 = nya
    else:
        nx1 = nx[0]
        ny1 = ny[0]
        nx2 = nx[1]
        ny2 = ny[1]
        print("Warning there are more than two notches being selected")
    
    return (nx,ny,nx1,ny1,nx2,ny2)
    

#############################################################################
def transrotateshape (nx1, ny1, nx2, ny2, nucx, nucy, cntx, cnty):
    """
    This takes all contour points and nuclei points and moves the shape so that
    the mid point of the two notches is now at (0,0). Then rotates shape so that
    the notches are on the x-axis.
    
    Requires
    --------
    pol2cart
    cart2pol
    
    Parameters
    ----------
    nx1, ny1, nx2, ny2: scalar
        Cartesian co-ordinates of notches
    nucx, nucy: array_like, shape(n,1)
        Cartesian co-ordinates of nuclei
    cntx, cnty: array_like, shape(n,1)
        Cartesian co-ordinates of contour points
    
    Returns
    --------
    tnx1, tny1, tnx2, tny2: scalar
        Cartesian co-ordinates of translated and rotated notches
    tnucx, tnucy: array_like, shape(n,1)
        Cartesian co-ordinates of translated and rotated nuclei
    tcntx, tcnty: array_like, shape(n,1)
        Cartesian co-ordinates of translated and rotated contour points
    
    Example
    --------
    #use arrays from outputs of above functions
    tnx1, tny1, tnx2, tny2, tnucx, tnucy, tcntx, tcnty = transrotateshape (1, 1, -1, -1, nucx, nucy, cntx, cnty)
    """
    
    """Find Notch Midpoint"""
    cx = (nx1+nx2)/2
    cy = (ny1+ny2)/2
    
    """Translation"""
    #Nuclei
    tnucx = nucx - cx
    tnucy = nucy - cy
    #Contour
    tcntx = cntx - cx
    tcnty = cnty - cy
    
    #Notches
    tnx1 = nx1 - cx
    tny1 = ny1 - cy
    tnx2 = nx2 - cx
    tny2 = ny2 - cy
    
    """Rotation"""
    #All points in polar co-ordinates
    nucrho, nucphi = cart2pol(tnucx,tnucy)
    cntrho, cntphi = cart2pol(tcntx,tcnty)
    n1rho, n1phi = cart2pol (tnx1, tny1)
    n2rho, n2phi = cart2pol(tnx2, tny2)
    
    #Rotation angle
    if tny1 > 0:
        rangle = n1phi
    else:
        rangle = n2phi
        
    #New phis
    nucphi = nucphi - rangle
    cntphi = cntphi - rangle
    n1phi = n1phi - rangle
    n2phi = n2phi - rangle
    
    #All points in Cartesian co-ordinates
    tnucx, tnucy = pol2cart(nucrho,nucphi)
    tcntx, tcnty = pol2cart(cntrho,cntphi)
    tn1x, tn1y = pol2cart(n1rho,n1phi)
    tn2x, tn2y = pol2cart(n2rho,n2phi)
    return (tn1x, tn1y, tn2x, tn2y, tnucx,tnucy,tcntx,tcnty)

#######################################################################################
def findNearestGreaterThan(searchVal, inputData):
    """
    This searches an array for the closest value greater than the value input,
    and returns the index of this value
    
    Parameters
    ----------
    searchVal: scalar
        Value to find closest value in the array 
    inputData: array_like, shape(n,1)
        Array to search
    
    
    Returns
    --------
    idx: scalar
        index of value closest to value input
    
    Example
    --------
    a = [0,2,4,6,8,10,12,14,16]
    b = findNearestGreaterThan(3,a)
    """
    import numpy as np
    diff = inputData - searchVal #subtract value from all
    diff[diff<0] = np.inf #if difference is less than zero set difference to infinity
    idx = diff.argmin() #find the index of least difference
    return idx

######################################################################################################
def findNearestLessThan(searchVal, inputData):
    """
    This searches an array for the closest value less than the value input,
    and returns the index of this value
    
    Parameters
    ----------
    searchVal: scalar
        Value to find closest value in the array 
    inputData: array_like, shape(n,1)
        Array to search
    
    
    Returns
    --------
    idx: scalar
        index of value closest to value input
    
    Example
    --------
    a = [0,2,4,6,8,10,12,14,16]
    b = findNearestLessThan(3,a)
    """
    import numpy as np
    diff = inputData - searchVal #subtract value from all
    diff[diff>0] = -np.inf #if difference is greater than zero set difference to negative infinity
    idx = diff.argmax() #find the index of least negative difference
    return idx


########################################################################################
def circularise(tnucx, tnucy, tcntx, tcnty, nucradii, nucbright):
    """
    This stretches every ray in the shape to create a circle.
    This uses elliptical fouriers to smooth the outline/provide additional
    points. Then it takes the contour polar co-ordinate rho (distance from 0,0)
    for every angle phi and divides all nuclear rhos with the same phi by that
    rho. 
    
    Requires
    ---------
    cart2pol
    pol2cart
    findNearestGreaterThan
    findNearestLessThan
    
    Parameters
    ----------
    tnucx, tnucy: array_like, shape(n,1)
        Cartesian co-ordinates of translated and rotated nuclei
    tcntx, tcnty: array_like, shape(n,1)
        Cartesian co-ordinates of translated and rotated contour points
    nucradii = aray_like, shape (n,1)
        Radii of found points
    nucbright = aray_like, shape (n,1)
        Brightness of found points
    
    Returns
    -------
    numnucoutcnt: scalar
        number of nuclei outside the given contour - they are placed on the contour
    cnucx, cnucy: array_like, shape(n,1)
        Cartesian co-ordinates of 'circularised' nuclei
    snucradii: array_like, shape(n,1) 
        Sorted list of nuclei radii so it matches the sorting of cnucx, and cnucy
    snucbright: array_like, shape(n,1)
        Sorted list of nuclei brightness so it matches the sorting of cnucx, and cnucy
    """
    
    """Import Functions"""
    import numpy as np
    
    """Increase phi angle as going down list"""
    
    elrho, elphi = cart2pol(tcntx,tcnty)
    for i in range(0,len(elphi)):
        if elphi[i] < elphi[0] or elphi[i] == elphi[len(elphi)-1]:
            elphi[i] = elphi[i] + np.pi*2
            
    #Making all nucphi positive - ensures that nucphi increases with increasing
    nucrho, nucphi = cart2pol(tnucx,tnucy)
    for i in range (0,len(nucphi)):
        if nucphi[i] < elphi[0] or nucphi[i] == elphi[len(elphi)-1]:
            nucphi[i] = nucphi[i] + np.pi*2
            
    nucphi = np.vstack(nucphi)
    nucrho = np.vstack(nucrho)
    nucradii = np.vstack(nucradii)
    nucbright = np.vstack(nucbright)
    
    """Sort nucphi smallest to largest and other descriptors"""
    #Create single array
    concat = np.append(nucphi,nucrho,axis= (1))
    concat = np.append(concat,nucradii,axis= (1))
    concat = np.append(concat,nucbright,axis= (1))
    
    #Sort array by nucphi
    concat = sorted(concat, key=lambda concat_entry: concat_entry[0])
    
    concat = np.asarray(concat)
    
    snucphi = concat[:,0]
    snucrho = concat[:,1]
    snucradii = concat[:,2]
    snucbright = concat[:,3]

    """Find Nearest contour point for each nuclei point"""
    for i in range(0,len(snucphi)):
        rhodiv = 0
        value = snucphi[i]
        maxind = findNearestGreaterThan(value, elphi)
        minind = findNearestLessThan(value, elphi)
        
        for j in range (minind,maxind+1):
            rho = elrho[j]
            if rho>rhodiv:
                rhodiv = rho
                
        if minind == len(elphi)-1:
            rho1 = elrho[minind]
            rho2 = elrho[0]
            rhodiv = max(rho1,rho2)
        if rhodiv == 0 or rhodiv == float('Inf'):
            rhodiv = 1
        snucrho1 = snucrho
        snucrho1[i] = snucrho1[i]/rhodiv
    """Any Without Contour Point"""
    subnucrho1 = []
    subnucphi1 = []
    indnucrho1 = []
    for i in range (0,len(snucrho)):
        if snucrho1[i]>1 and not snucrho1[i]==  float('Inf'):
            subnucrho1 = np.append(subnucrho1,snucrho[i])
            indnucrho1 = np.append(indnucrho1,i)
            subnucphi1 = np.append(subnucphi1,snucphi[i])
    subnucphi1 = np.round(subnucphi1,0)
    uniquephi = np.array(list(set(subnucphi1)))
    for i in range (0,len(uniquephi)):
        rhodiv = 1
        itemindex = np.where(subnucphi1==uniquephi[i])
        for j in range (0, len(itemindex[0])):
            ind = itemindex[0][j]
            rho = subnucrho1[int(ind)]
            if rho>rhodiv:
                rhodiv = rho
        for j in range (0, len(itemindex[0])):
            ind = int(itemindex[0][j])
            index = int(indnucrho1[ind])
            snucrho1[index] = subnucrho1[ind]/(rhodiv)
            if snucrho1[index]>1:
                snucrho1[index] = 1
    snucrho = snucrho1
    #convert back to cartesian co-ordinates
    cnucx, cnucy = pol2cart(snucrho,snucphi)
    return(cnucx,cnucy,snucradii, snucbright)


###############################################################################################


def outputplots(cnucx,cnucy,snucradii, snucbright, verbosity = True, imagename = None, sctplt = False, sizeplt = False, heatblockplt = False, heatmapplt = False ):
    """
    Four different possible plots displaying circularised nuclei data:
        1: Scatter plot - shows location of circularised nuclei and circle outline
        2: Size plot - shows nuclei size and brightness as well as location and a unit circle
        3: Heat block plot - shows a nuclei density map using blocks
           with 900 blocks and limits of -1 and 1 for x and y direction
        4: Heat map plot - shows a heat map - smoother than Heat block plot - limits of -1 and 1 for x and y direction
    
    Requires
    ---------
    circles
    
    Parameters
    ----------
    cnucx, cnucy: array_like, shape(n,1)
        Cartesian co-ordinates of 'circularised' nuclei
    snucradii: array_like, shape(n,1) 
        Sorted list of nuclei radii so it matches the sorting of cnucx, and cnucy
    snucbright: array_like, shape(n,1)
        Sorted list of nuclei brightness so it matches the sorting of cnucx, and cnucy
    verbosity: Boolean, Default: True
        If True all plots are saved
    imagename: String, Default:None
        If saving this is required as the basis of the name to save with.
        It should end in a three digit sequence number and .tif
    sctplt: Boolean, optional, Default:False
        If true a scatter plot is produced
    sizeplt: Boolean, optional, Default:False
        If true a size plot is produced
    heatblockplt: Boolean, optional, Default:False
        If true a heat block plot is produced
    heatmapplt: Boolean, optional, Default:False
        If true a heat map plot is produced
    """
    
    """Import Functions"""
    import numpy as np    
    import pylab
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage.filters import gaussian_filter
    
    """Create Name for Output Images"""
    if verbosity:    
        #Create new image name
        pointind = imagename.find('.')
        pointindmin3 = pointind-3
        seriesname = imagename[pointindmin3:pointind]
        nucpointimname = imagename[:pointindmin3]
        #nucpointimname = nucpointimname+seriesname + ".tif"
    
    """Scatter Plot"""
    if sctplt:
        #Create circle contour
        t = np.linspace(0,2*np.pi, 360)
        
        xi = [0]
        yi = [1]
        
        for i in range(0, len(t)):
            xi = np.vstack([xi, np.cos(t[i])])
            yi = np.vstack([yi, np.sin(t[i])])
        
        plt.figure()
        pylab.figure(figsize=(10,10))
        plt.scatter(cnucx,cnucy)
        plt.scatter(xi,yi)
        plt.gca().set_aspect('equal', adjustable='box')
        
        if verbosity:
            plt.savefig(nucpointimname +"_Sctr_"+seriesname + ".tif", bbox_inches='tight')
    
    """Size Plot"""
    if sizeplt:
        #Create Figure
        
        pylab.figure(figsize=(20,15))
        
        #Reshape Brightness for function use
        #used as there were some zeros and logarithmic
        #scale so variation can better be seen
        b = snucbright.reshape(-1)
        b = np.log(b+1) 
        
        #Plot circles
        #plot a set of circles
        out = circles(cnucx, cnucy, snucradii/300, c=b, alpha=.6, ec='none')
        circles(0,0,1,c=b, alpha = 0.1, ec = 'none') #background circle
        plt.colorbar(out)
        if verbosity:
            plt.savefig(nucpointimname +"_Size_"+seriesname + ".tif", bbox_inches='tight')
    
    """Heat Block Plot"""
    if heatblockplt: 
        #reshape input
        x = cnucx.reshape(-1)
        y = cnucy.reshape(-1)
        
        #create heatmap
        xedges = np.asarray([ -9.93030466e-01,  -9.26782368e-01,  -8.60534270e-01,
        -7.94286172e-01,  -7.28038074e-01,  -6.61789976e-01,
        -5.95541878e-01,  -5.29293780e-01,  -4.63045682e-01,
        -3.96797584e-01,  -3.30549486e-01,  -2.64301389e-01,
        -1.98053291e-01,  -1.31805193e-01,  -6.55570948e-02,
         6.91003109e-04,   6.69391010e-02,   1.33187199e-01,
         1.99435297e-01,   2.65683395e-01,   3.31931493e-01,
         3.98179591e-01,   4.64427689e-01,   5.30675786e-01,
         5.96923884e-01,   6.63171982e-01,   7.29420080e-01,
         7.95668178e-01,   8.61916276e-01,   9.28164374e-01,
         9.94412472e-01])
        yedges= np.asarray([-0.97497741, -0.91135149, -0.84772557, -0.78409965, -0.72047373,
       -0.65684782, -0.5932219 , -0.52959598, -0.46597006, -0.40234414,
       -0.33871823, -0.27509231, -0.21146639, -0.14784047, -0.08421455,
       -0.02058864,  0.04303728,  0.1066632 ,  0.17028912,  0.23391503,
        0.29754095,  0.36116687,  0.42479279,  0.48841871,  0.55204462,
        0.61567054,  0.67929646,  0.74292238,  0.8065483 ,  0.87017421,
        0.93380013])
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[xedges,yedges])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        #plot figure
        plt.figure()
        pylab.figure(figsize=(20,15))
        im = plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.colorbar(im)
        if verbosity:
            plt.savefig(nucpointimname +"Htblck_"+seriesname + ".tif", bbox_inches='tight')
            
    """Heat Map Plot"""
    if heatmapplt:
        #Reshape input
        x = cnucx.reshape(-1)
        y = cnucy.reshape(-1)
        
        #Create heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=1500)
        heatmap = gaussian_filter(heatmap, sigma=19)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        #Plot Figure
        plt.figure()
        pylab.figure(figsize=(20,15))
        im = plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet) 
        plt.colorbar(im)
        if verbosity:
            plt.savefig(nucpointimname +"_HtMp_"+seriesname + ".tif", bbox_inches='tight')
    return()

######################################################################
def nucleioverimage(imname, nucx, nucy, nucradii, nucbright, savefig = True, savefigname = None):
    """
    Plot nuclei (location, size and brightness) over an image.
    
    Requires
    ---------
    circles
    
    Parameters
    ----------
    imname: string
        Include file location, name of image to overlay
    savefig: Boolean, Default:True
        If true then the figure made is saved
    savefigname: string
        Name with which to save the figure
    nucx, nucy: array_like, shape(n,1)
        Cartesian co-ordinates of nuclei
    nucradii: array_like, shape(n,1)
        Nuclei radii
    nucbright: array_like, shape(n,1)
        Nuclei brightness/intensity
    
    """
    #Import functions
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    #Create figure
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    
    """Create background image"""
    fname = imname
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    plt.imshow(arr, cmap='gray', alpha = .4)
    
    """Create nuclei overlay"""
    #Reshape Brightness for function use
    #used as there were some zeros and logarithmic
    #scale so variation can better be seen
    b = nucbright.reshape(-1)
    b = np.log(b+1) 
        
    #Plot circles
    out = circles(nucx, 1024 - nucy, nucradii, c=b, alpha=.4, ec='none')
    plt.colorbar(out)
    
    #Plot figure

    if savefig:
        fig.savefig(savefigname, dpi=1024)
    plt.show()
    return

