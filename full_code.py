# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:08:19 2021

@author: Matth
"""

#generate some models with X paramters, save those parameters in X file
#scan those models 
#popout the top 5% closest coverage ones 

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:24:54 2021

@author: Matth
"""

#libraries
import numpy as np; import itertools;
import pandas as pd
import math
from math import sqrt
import matplotlib.pyplot as plt
import time
import glob, os
from skimage import data
from skimage.color import rgb2gray
from skimage import measure
import numpy as np; import matplotlib.pyplot as plt; import seaborn as sb;
import time; import random as r; import pandas as pd; import math

#define a function that takes a random length and width
#inputs are length, width, graphene#, top left coord, starting points
#that function should return an array of a randomly generated shape

def generateShape(length, width, layer, maxLayer, right, top, left, bottom, grArray, patchY, patchX):
    shape = np.zeros((length, width))+maxLayer;  #y rows, x columns
    #create an array to store contour points
    contoursY = np.zeros((length+width)*2+10, dtype=int)
    contoursX = np.zeros((length+width)*2+10, dtype=int)
    contourCounter = 0
    startY = 0
    startX = 0
    endY = 0
    endX = 0
    for i in range(4):
        #print(i)
        if(i==0): #if going from right to top
            startY = right
            startX = width-1
            endY = 0
            endX = top
            contoursY[0], contoursX[0] = startY, startX
        elif(i==1): #if going from top to left
            startY = 0
            startX = top
            endY = left
            endX = 0
        elif(i==2): #if going from left to bottom
            startY = left
            startX = 0
            endY = length-1
            endX = bottom
        else: #if going from bottom to right
            startY = length-1
            startX = bottom
            endY = right
            endX = width-1
        endFlag = True
        while(endFlag):
            #determine walking direction
            contourCounter +=1
            if(contourCounter >= ((length+width)*2+10)):
                print("out of bounds")
            if(startY>endY):
                directionY = -1
            elif(startY<endY):
                directionY = 1
            else:
                directionY = 0
            if(startX>endX):
                directionX = -1
            elif(startX<endX):
                directionX = 1
            else:
                directionX = 0    
            #randomly select walking direction
            direction = round(np.random.rand()); #create random number. If 0, change y. If 1, change x.
            
            l = len(grArray)
            w = len(grArray[0])
            if((contoursY[contourCounter-1] + directionY + patchY) < l):
                if(grArray[(contoursY[contourCounter-1] + directionY + patchY),contoursX[contourCounter-1]+patchX] == maxLayer):
                    pass;
                else:
                    direction = 1;
            if((contoursX[contourCounter-1] + directionX + patchX) < w):
                if(grArray[contoursY[contourCounter-1]+patchY,contoursX[contourCounter-1] + directionX + patchX] == maxLayer):
                    pass;
                else:
                    direction = 0;
                
            #prevent from going outside boundingbox or passing last value
            if(contoursY[contourCounter-1] == endY):
                if(contoursX[contourCounter-1] == endX):
                    contourCounter -=1;
                    endFlag = False
                    break;
                else:
                    direction = 1
            elif(contoursX[contourCounter-1] == endX):
                if(contoursY[contourCounter-1] == endY):
                    contourCounter -=1;
                    endFlag = False
                    break;
                else:
                    direction = 0        
            else:
                pass
            #walk from one point to the next
            #move y
            if(direction == 0):
                contoursY[contourCounter] = contoursY[contourCounter-1] + directionY
                contoursX[contourCounter] = contoursX[contourCounter-1]
                if(contoursY[contourCounter] < 0 or contoursX[contourCounter] < 0):
                    print("less than 0");
            #move x
            else:
                contoursX[contourCounter] = contoursX[contourCounter-1] + directionX
                contoursY[contourCounter] = contoursY[contourCounter-1]
                if(contoursY[contourCounter] < 0 or contoursX[contourCounter] < 0):
                    print("less than 0");
    #fills in the contours on the graphene array
    for i in range(contourCounter):
        shape[contoursY[i],contoursX[i]] = layer
    #fill in the shape
    #first scan a given y range for the number of layer changes
    #after first encounter with layer you need one layer change to 0 layers and 1 layer change to then fill in
    #after the 1 layer change, record that as initial point
    #then record the ending point
    #then fill in
    for j in range(length):
        initialFlag = False;
        startFlag = False;
        changeStartCoord = 0;
        endFlag = False;
        changeEndCoord = 0;
        for i in range(width):
            if(shape[j][i] == layer and initialFlag == False):
                initialFlag = True;
            if(initialFlag == True and startFlag == False and shape[j][i] == maxLayer):
                changeStartCoord = i;
                startFlag = True;
            if(startFlag == True and endFlag == False and shape[j][i] == layer):
                changeEndCoord = i;
                endFlag = True;
        if(endFlag == True):
            for i in range(changeStartCoord, changeEndCoord):
                shape[j][i] = layer;
    return shape;
#define a function that scans through an array and outputs:
def checkScan(grArray, length, width, maxLayer, y, x):
    coverageFraction = 0;
    right = np.zeros(length)
    rightCounter = 0;
    left = np.zeros(length)
    leftCounter = 0;
    top = np.zeros(width)
    topCounter = 0;
    bottom = np.zeros(width)
    bottomCounter = 0;
    freeCounter = 0;
    for j in range(y, y+length):
        for i in range(x, x+width):
            if(grArray[j][i] == maxLayer):
                freeCounter+=1;
                if(j == y):
                    top[topCounter] = (i-x)
                    topCounter+=1
                if(j == (y+length-1)):
                    bottom[bottomCounter] = (i-x)
                    bottomCounter+=1
                if(i == x):
                    left[leftCounter] = (j-y)
                    leftCounter+=1
                if(i == (x+width-1)):
                    right[rightCounter] = (j-y)
                    rightCounter+=1
    total=length*width
    coverageFraction = 1-(freeCounter/total)
    returnArray = [coverageFraction, right, top, left, bottom]            
    return returnArray
def calculateCoverage(grArray, length, width, maxLayer):
    coverage = np.zeros(maxLayer+1);
    for j in range(length):
        for i in range(width):
            num = int(grArray[j,i])
            coverage[num] += 1/(length*width);
    return coverage;
    #the number of graphene
    #the number of nongraphene
    #the coverage fraction
    #an array of points on the right, left, top, and bottom walls that are free
    #the array of values
#create a graphene array of certain dimensions
time0 = time.time();

#change this to the folder where you're keeping everything
path=os.getcwd()+'/Demo/'
dfinput = pd.read_csv(path+'Generation/'+'input.csv')
layers = (len(dfinput.loc[:,'coverage']))-1
dfParameters = pd.read_csv(path+'Scanning/'+'parameters.csv')

# Input parameters: Sample dimensions, beam radius, optical step length, all in meters
l= float(dfinput.loc[0,'length']); #rows, vertical, up-down
w= float(dfinput.loc[0,'width']); #collumns, horizontal, left-right
beamRadius = float(dfinput.loc[0,'beamRadius']);
opticalStepLength = float(dfinput.loc[0,'opticalStepLength']);

convergence = False;
numConverged = int(dfinput.loc[0,'numConverged']);

while(not(convergence)):
    
    # Input parameters for random graphene placement
    acceptableOverlapParam = float(dfinput.loc[0,'acceptableOverlap']);    # acceptable amount of overlap (per patch)
    charPatchLParam = float(dfinput.loc[0,'charLengthL']);           # mean characteristic patch length
    charPatchLParamVar = float(dfinput.loc[0,'charLengthLVAR']);           # mean characteristic patch length variance
    charPatchLVarParam = float(dfinput.loc[0,'charLengthVarL']);        # characteristic patch length variance
    charPatchLVarParamVar = float(dfinput.loc[0,'charLengthVarLVAR']);        # characteristic patch length variance
    charPatchWParam = float(dfinput.loc[0,'charLengthW']);           # mean patch width
    charPatchWParamVar = float(dfinput.loc[0,'charLengthWVAR']);           # mean patch width
    charPatchWVarParam = float(dfinput.loc[0,'charLengthVarW']);        # patch width variance
    charPatchWVarParamVar = float(dfinput.loc[0,'charLengthVarWVAR']);        # characteristic patch length variance
    coverageTolerance = 0.05;
    
    subtractiveMode = True;     # boolean variable to turn on subtractive mode
    # initial number of layers in subtractive mode
    size = layers+1;
    desiredCoverageParam = [0]*(size);      # fraction of sample covered by graphene
    #sets coverage fraction for each layer
    desiredCoverageParamVar = [0]*(size);
    for i in range(size): #set the coverage for each layer using user input
        desiredCoverageParam[i] = float(dfinput.loc[i,'coverage']);
        desiredCoverageParamVar[i] = float(dfinput.loc[i,'coverageVar']);
    # Calculate beam sigma. 
    sigma = 2*beamRadius;
    #Calculate dimensions of final beam power array
    powerDimY = math.ceil(l/opticalStepLength);
    powerDimX = math.ceil(w/opticalStepLength);
    #sample Big, powerSmall
    # Calculate dimensions for sample array based on beam power array where the sample array is higher resolution
    resRatio = 5;
    sampleDimY = powerDimY*resRatio;
    sampleDimX = powerDimX*resRatio;
    sampleArea = sampleDimX*sampleDimY; 
    pixelSize = opticalStepLength*resRatio;
    pixelArea = pixelSize*pixelSize;
    
    #model generating parameters
    numModels = int(dfinput.loc[0,'numModels']);
    charLengthLVector = np.zeros(numModels);
    charLengthWVector = np.zeros(numModels);
    charLengthVarianceWVector = np.zeros(numModels);
    charLengthVarianceLVector = np.zeros(numModels);
    coverageVector = np.zeros((numModels,size));
    # START DANNY'S CODE
    for f in range(numModels):
        if(os.path.isfile(path+'model'+str(f)+'.csv')):
            charLengthWVector[f] = dfParameters.loc[f,'charLengthW'];
            charLengthLVector[f] = dfParameters.loc[f,'charLengthL'];
            charLengthVarianceWVector[f] = dfParameters.loc[f,'charLengthVarW'];
            charLengthVarianceLVector[f] = dfParameters.loc[f,'charLengthVarL'];
            coverage = np.zeros(size);
            for i in range(size):
                coverage[i] = dfParameters.loc[f,('coverage'+str(i))]
            coverageVector[f] = coverage;
            continue;
        else:
            pass;
        #Choose Characteristic Length
        #Choose Characteristic Variance
        desiredCoverage = [0]*(size)
        for i in range(size):
            #Choose Coverage
            desiredCoverage[i] = abs(np.random.normal(desiredCoverageParam[i], abs(desiredCoverageParamVar[i])));      # fraction of sample covered by graphene
        acceptableOverlap = abs(acceptableOverlapParam);    # acceptable amount of overlap (per patch)
        charPatchLVar = abs(np.random.normal(charPatchLVarParam, abs(charPatchLVarParamVar)));        # characteristic patch length variance
        charPatchL = abs(np.random.normal(charPatchLParam, abs(charPatchLParamVar)));# mean characteristic patch length
        charPatchWVar = abs(np.random.normal(charPatchWVarParam, abs(charPatchWVarParamVar)));
        charPatchW = abs(np.random.normal(charPatchWParam, abs(charPatchWParamVar)));# mean patch width
    # Generate Graphene Array
        grArray = np.zeros((sampleDimY,sampleDimX));  #y rows, x columns
        coverage = [0]*(size); # initial coverage fraction is 0
    #randomly select graphene patch in a for loop for N locations
        grArray += (layers);
        coverage[layers] = 1; #sets coverage fraction to 1 for max layers
        for h in range(layers, -1, -1): #do this for each layer
            numPatches = 0;
            # iterate until "desiredCoverage" of graphene is met
            coverageFlag=0;
            timenow=0
            while(coverage[h]<(desiredCoverage[h]-coverageTolerance)):
                overlapFlag = True;
                print("coverage:" + str(coverage));
                attempts = 0; # instantiate variable for attempts at placing a patch and sucessful placements
                while(overlapFlag):
    # increment attempts 
                    attempts += 1;
                    print("attempts:" + str(attempts));
    # randomly select index pair in sample array as top left of new patch
                    patchY = int(r.random()*sampleDimY);
                    patchX = int(r.random()*sampleDimX);
    # randomly select a patch length and width
                    patchL = abs(int((np.random.normal(charPatchL, charPatchLVar))/(opticalStepLength/resRatio)));
                    patchW = abs(int((np.random.normal(charPatchW, charPatchWVar))/(opticalStepLength/resRatio)));
                    print("patchL: " + str(patchL) + " patchW: " + str(patchW))
                    if(patchL < 5): #minimum feature size
                        patchL = 5;
                    if(patchW < 5):
                        patchW = 5; #minimum feature size
                    patchArea = patchL*patchW;
    #maximum Feature SIze
                    if(patchL > sampleDimY): #minimum feature size
                        patchL = sampleDimY;
                    if(patchW > sampleDimX):
                        patchW = sampleDimX; #minimum feature size
    # constrain patch to sample area
                    if ((patchY+patchL) > sampleDimY):
                        offset = (patchY+patchL)-(sampleDimY)
                        patchY = patchY-offset
                    else:
                        pass;
                    if ((patchX+patchW) > sampleDimX):
                        offset = (patchX+patchW)-(sampleDimX)
                        patchX = patchX-offset
                    else:
                        pass;
    #if overlap < required overlap, then break and choose that one, otherwise use smallest of N 
                    tempArray = checkScan(grArray, patchL, patchW, layers, patchY, patchX)
                    if(tempArray[0] < acceptableOverlap):
                        overlapFlag = False;
                        break;
                    if(attempts > 10):
                        overlapFlag = False;
                        break;
                right = tempArray[1][math.floor(np.random.rand()*tempArray[1].size)]
                top = tempArray[2][math.floor(np.random.rand()*tempArray[2].size)]
                left = tempArray[3][math.floor(np.random.rand()*tempArray[3].size)]
                bottom = tempArray[4][math.floor(np.random.rand()*tempArray[4].size)]
                tempArrayGraph = generateShape(patchL, patchW, h, layers, right, top, left, bottom, grArray, patchY, patchX)
                grArray[patchY:(patchY+patchL),patchX:(patchX+patchW)] = tempArrayGraph
                coverage = calculateCoverage(grArray, sampleDimY, sampleDimX, layers);
                numPatches+=1
                print("num patches: " + str(numPatches))
    #TRANSMISSION                    
            
        # Calculate transmission through N layers of graphene using alpha = fine structure constant
        alpha = 1/137;
        sampleArray = (1+1.13/2*np.pi*alpha*grArray)**-2;
        
        sb.heatmap(np.transpose(sampleArray), square=True); plt.title('input pattern'); plt.show()
        plt.hist(sampleArray.flatten());plt.xlabel('transmission'); plt.ylabel('counts');
        plt.title('Histogram of input pattern'); plt.show()
        
        time1 = time.time();
        
        print('total time = ' + str(time1 - time0))
        #save to csv NEED TO MODIFY TO INCLUDE parameter data
        #maybe just include another CSV file that has parameter data in it for simplicity
        df = pd.DataFrame(data=grArray) #change this back to transmission array
        iterationNumber = str(f)
        print('iteration: '+ iterationNumber)
        df.to_csv(path+'Generation/'+'model'+iterationNumber+'.csv',index=False)
        charLengthWVector[f] = charPatchW;
        charLengthLVector[f] = charPatchL;
        charLengthVarianceWVector[f] = charPatchWVar;
        charLengthVarianceLVector[f] = charPatchLVar;
        coverageVector[f] = coverage;
    dataDictionary = {'charLengthW':charLengthWVector, 'charLengthVarW':charLengthVarianceWVector, 'charLengthL':charLengthLVector, 'charLengthVarL':charLengthVarianceLVector}
    for i in range(size):
        dataDictionary['coverage'+str(i+1)] = coverageVector[:,i];
    dfParameters = pd.DataFrame(data=dataDictionary);
    dfParameters.to_csv(path+'Scanning/'+'parameters.csv');    
    
    # This code simulates our spatially resolved optical transmission experiment
    
    def beamWeight(x0,y0,coords):
        # (modify this to potentially become a lambda equation, may reduce calculation time)
        d = np.sqrt(np.sum(abs(np.subtract((x0,y0),coords)*pixelSize[0]),axis=1))
        # p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }} e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} }
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-d**2/(2*sigma**2))
    
    time0 = time.time();
    
    # Calculate beam sigma. Calculate dimensions of final beam power array
    powerDim = math.ceil(l/opticalStepLength), math.ceil(w/opticalStepLength);
    
    # Calculate dimensions for sample array based on beam power array where the sample array is higher resolution
    n,m = powerDim[0]*resRatio, powerDim[1]*resRatio;
    arrayDim = int(powerDim[0]*resRatio), int(powerDim[1]*resRatio);
    pixelSize = l/arrayDim[0], w/arrayDim[1];
    pixelArea = pixelSize[0]*pixelSize[1]
    
    #import from pandas
    #imports models from CSVs's
    Mod=[]
    numModels = 0
    os.chdir(path +"Generation/")
    files = glob.glob("*.csv")
    for a in files:
        if ('model' in a):
            numModels+=1
            print(a)
            Mod.append(pd.read_csv(path+"Generation/"+a))
        else:
            pass
    print("numModels = " + str(numModels))
        
    for f in range(numModels):
            
        # Calculate transmission through N layers of graphene using alpha = fine structure constant
        alpha = 1/137;
        grArray = Mod[f].to_numpy()
        sampleArray = (1+1.13/2*np.pi*alpha*grArray)**-2;
        
        sb.heatmap(sampleArray); plt.title('input pattern'); plt.show()
        plt.hist(sampleArray.flatten());plt.xlabel('transmission'); plt.ylabel('counts');
        plt.title('Histogram of input pattern'); plt.show()
        
        # Optical step length is converted to number of pixels via resolution ratio, ie how much higher in resolution
        # the sample array will be compared to the final power array
        stepPixelsL = int(resRatio);
        stepPixelsW = int(resRatio);
        
        # Place beam centers to be considered based on optical step length in pixels in flattened arrays
        x,y = np.meshgrid(np.arange(0,arrayDim[0],stepPixelsL),np.arange(0,arrayDim[1],stepPixelsW));
        xCenters = x.ravel(); yCenters = y.ravel();
        
        # Distance to be considered during power summation, ie 3 sigma
        nSigma = 3;
        pixelnSigmaL = int(nSigma*sigma/pixelSize[0]);
        pixelnSigmaW = int(nSigma*sigma/pixelSize[1]);
        
        # Iterate over beam center values
        transmissionArray = []; temp=np.zeros(powerDim); 
        for i, x0 in enumerate(xCenters):
            y0 = yCenters[i];
            # For each beam center, convert to distance location on sample assuming beam is a rectangle
            xIdx = np.arange(x0 - pixelnSigmaL, x0 + pixelnSigmaL,1);
            yIdx = np.arange(y0 - pixelnSigmaW, y0 + pixelnSigmaW,1);
            
            # Check if pixel range is within sample boundaries. If necessary, remove indices beyond sample edges
            if np.logical_or(xIdx < 0,yIdx < 0).sum() > 0:
                fixedIdx = np.argwhere(np.logical_or(xIdx < 0,yIdx < 0));
                xFixedIdx = xIdx[int(fixedIdx[-1]) + 1:]; yFixedIdx = yIdx[int(fixedIdx[-1]) + 1:];
            else:
                xFixedIdx = xIdx; yFixedIdx = yIdx;
            if np.logical_or(xFixedIdx > arrayDim[0] - 1, yFixedIdx > arrayDim[1] - 1).sum() > 0:
                fixedIdx = np.argwhere(np.logical_or(xFixedIdx > arrayDim[0] - 1, yFixedIdx > arrayDim[1] - 1));
                xFixedIdx = xFixedIdx[:int(fixedIdx[0]) - 1]; yFixedIdx = yFixedIdx[:int(fixedIdx[0]) - 1]; 
            
            # Convert final indices and beam center location to distances/coordinates on sample array
            xCoord = xFixedIdx*pixelSize[0]; yCoord = yFixedIdx*pixelSize[1];
            x0Coord = x0*pixelSize[0]; y0Coord = y0*pixelSize[1];
            
            # Place distance coordinates in tuple for faster calculation (maybe)
            coords = tuple(itertools.product(xCoord,yCoord));
            xCoordsIdx, yCoordsIdx = np.meshgrid(xFixedIdx,yFixedIdx);
        
            # Get power values at all pixels within n*sigma of beam center
            basePower = beamWeight(x0Coord,y0Coord,coords);
            basePower = np.reshape(basePower,(len(xFixedIdx),len(yFixedIdx)));
            
            # Get sample transmissions to use for power adjustment based on graphene's presence
            sampleSub = sampleArray[xCoordsIdx,yCoordsIdx];
            # Multiple base power by sample transmissions to get measured power. Sum all measured power values in area
            # and divide by sum of all base power values in area to get area transmission
            measuredPower = basePower*sampleSub;
            transmission = measuredPower.sum()/basePower.sum();
            transmissionArray.append(transmission);
            
        # Show heatmap and hist of detected power array as normalized by max power value
        transmissionArray = np.asarray(transmissionArray);
        transmissionArray = np.reshape(transmissionArray,powerDim).T;
        
        sb.heatmap(transmissionArray); plt.show()
        plt.hist(transmissionArray.flatten()); plt.xlabel('transmission'); plt.ylabel('counts'); 
        plt.title('Histogram post measurement'); plt.show()
        
        time1 = time.time();
        
        print('total time = ' + str(time1 - time0))
        
        df = pd.DataFrame(data=transmissionArray) #change this back to transmission array
        iterationNumber = str(f)
        print('iteration: '+ iterationNumber)
        df.to_csv(path+'Scanning/'+'model'+iterationNumber+'.csv',index=False)
        
    #import experimental heatmap (likelihood)
    #B = distribution of graphene
    #A(i) = some parameters (characteristic length, overlap, etc.)
    #B|A 
    dfEx = pd.read_csv(path+'Scanning/'+'experimental.csv')
    dfEx = dfEx.fillna(method='backfill')
    dfParameters = pd.read_csv(path+'Scanning/'+'parameters.csv')
    layers = (len(dfParameters.columns))-5
    #imports models from CSVs's
    dfMod=[]
    numModels = 0
    os.chdir(path+'Scanning/')
    files = glob.glob("*.csv")
    for a in files:
        if ('model' in a):
            numModels+=1
            print(a)
            dfMod.append(pd.read_csv(path+'Scanning/'+a))
        else:
            pass
    print(numModels);
    
    #sets rejection rate of ABC
    error = dfinput.loc[0,'error']
    #numModels Saved
    numSaved = math.ceil(numModels*error);
    
    #create an array of Models with their Dist values and Coverage %
    dfDistMod = pd.DataFrame(np.zeros((numModels,9)),columns=['Path_Coverage','charLengthW','Path_charLengthW','charLengthSTDW','Path_charLengthSTDW','charLengthL','Path_charLengthL','charLengthSTDL','Path_charLengthSTDL'])
    for i in range(layers):
        string = 'coverage' + str(i+1)
        dfDistMod[string] = dfParameters.loc[:,string]
    dfDistMod.loc[:,'charLengthW'] = dfParameters.loc[:,'charLengthW']
    dfDistMod.loc[:,'charLengthSTDW'] = dfParameters.loc[:,'charLengthVarW'].pow(0.5)
    dfDistMod.loc[:,'charLengthL'] = dfParameters.loc[:,'charLengthL']
    dfDistMod.loc[:,'charLengthSTDL'] = dfParameters.loc[:,'charLengthVarL'].pow(0.5)
    
    def coverageCheck(dfEx, dfMod):#import experimental and mod array, return an array of pathcoverage
        #Get Path/RMS for each model compared to experimental
        #sort experiment by rows and columns 
        dfExSorted = pd.DataFrame(np.sort(dfEx.values, axis=0), index=dfEx.index, columns=dfEx.columns)
        dfExSorted = pd.DataFrame(np.sort(dfExSorted.values, axis=1), index=dfExSorted.index, columns=dfExSorted.columns)
        #sort model by rows and columns
        dfModSorted=[]
        distCoverage = np.zeros(numModels)
        for i in range(numModels):
            dfModSorted.append(pd.DataFrame(np.sort(dfMod[i].values, axis=0), index=dfMod[i].index, columns=dfMod[i].columns))
            dfModSorted[i] = pd.DataFrame(np.sort(dfModSorted[i].values, axis=1), index=dfModSorted[i].index, columns=dfModSorted[i].columns)
            #RMS (Subtract DFE by each DFM; Square all new values; Sum all new values into new DFM)
            distCoverage[i] = (np.sum(np.abs(np.subtract(dfModSorted[i].to_numpy(), dfExSorted.to_numpy()))))/np.sum(dfExSorted.to_numpy());
        return distCoverage
    #sort by Coverage
    dfDistMod.loc[:,'Path_Coverage'] = coverageCheck(dfEx, dfMod);
    
    def sortDistance(dfDistMod, collumn): #takes in dataframe with distance functions, and sorts them by specified collumn
        #sort by RMS (least to greatest) & pop out top 5%.
        dfDistSorted = pd.DataFrame(dfDistMod.sort_values(by=[collumn]))
        print(dfDistSorted)
        return dfDistSorted
    
    
    def charLengthCheck(r, layer, dfEx, dfMod): #input: r value, layer; output: charLengthW, charLengthL, charLengthWVar, charLengthLVar 
        #Characteristic Length
        minLayer = layer
        maxLayer = layer+1;
        alpha = 1/137;
        minVal = (1+1.13/2*np.pi*alpha*maxLayer)**-2;
        maxVal = (1+1.13/2*np.pi*alpha*minLayer)**-2;
        dfEx_grayScale = dfEx.to_numpy()
        dfEx_grayScale = dfEx_grayScale.copy(order='C')
        tolerance = 0.001
        dfEx_grayScale = np.where((dfEx_grayScale > (minVal-tolerance)) & (dfEx_grayScale < (maxVal-tolerance)), 0, 1)
        # Find contours at a constant value of r
        contours_Ex = measure.find_contours(dfEx_grayScale, level=r, fully_connected='high')
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(dfEx_grayScale, cmap=plt.cm.gray)
        
        numContoursEx = len(contours_Ex)
        lengthEx = np.zeros(numContoursEx)
        widthEx = np.zeros(numContoursEx)
        contourCounter = 0
        for contour in contours_Ex:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            #print(contourCounter)
            lengthEx[contourCounter] = np.amax(contours_Ex[contourCounter][:, 0]) - np.amin(contours_Ex[contourCounter][:, 0])
            widthEx[contourCounter] = np.amax(contours_Ex[contourCounter][:, 1]) - np.amin(contours_Ex[contourCounter][:, 1])
            contourCounter+=1
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
        dfMod_grayScale = []
        
        dfMod_charLength = np.zeros([numModels, 5], dtype=object) #create array for length, width, varlength, var width
        
        for i in range(numModels):
            #Characteristic Length
            dfMod_grayScale.append(dfMod[i].to_numpy())
            dfMod_grayScale[i] = dfMod_grayScale[i].copy(order='C')
            dfMod_grayScale[i] = np.where((dfMod_grayScale[i] > (minVal-tolerance)) & (dfMod_grayScale[i] < (maxVal-tolerance)), 0, 1)
            # Find contours at a constant value of 0.8
            contours_Mod = measure.find_contours(dfMod_grayScale[i], level=r, fully_connected='high')
            #RMS (Subtract DFE by each DFM; Square all new values; Sum all new values into new DFM)
            #plot characteristic length for check
            fig, ax = plt.subplots()
            ax.imshow(dfMod_grayScale[i], cmap=plt.cm.gray)
            
            dfMod_charLength[i, 0] = len(contours_Mod) #numContours
            numContours = int(dfMod_charLength[i,0])
            dfMod_charLength[i, 1] = np.zeros(numContours) #length
            dfMod_charLength[i,2] = np.zeros(numContours) #width
            contourCounter = 0  
            
            for contour in contours_Mod:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
                dfMod_charLength[i, 1][contourCounter] = np.amax(contours_Mod[contourCounter][:, 0]) - np.amin(contours_Mod[contourCounter][:, 0])
                dfMod_charLength[i, 2][contourCounter] = np.amax(contours_Mod[contourCounter][:, 1]) - np.amin(contours_Mod[contourCounter][:, 1])
            contourCounter+=1
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()
            
            #divide sum of lengths by number of contours to get distance function for charLength
            lengthDistanceEx = (np.sum(lengthEx))/len(contours_Ex)
            #divide sum of widths by number of contours to get distance function for charWidth
            widthDistanceEx = (np.sum(widthEx))/len(contours_Ex)
            #compute variance of Lengths
            lenVarDistanceEx = np.std(lengthEx)
            #compute variance of Widths
            widVarDistanceEx = np.std(widthEx)
            
        #repeat for models
        for i in range(numModels):
            #Characteristic Length
            dfMod_charLength[i, 3] = np.std(dfMod_charLength[i,1])
            dfMod_charLength[i, 4] = np.std(dfMod_charLength[i,2])
            dfMod_charLength[i,1] = (np.sum(dfMod_charLength[i,1]))/dfMod_charLength[i,0]
            dfMod_charLength[i,2] = (np.sum(dfMod_charLength[i,2]))/dfMod_charLength[i,0]
            #turn to distance function
            dfMod_charLength[i,1] = abs(dfMod_charLength[i,1]-lengthDistanceEx)/lengthDistanceEx
            dfMod_charLength[i,2] = abs(dfMod_charLength[i,2]-widthDistanceEx)/widthDistanceEx
            dfMod_charLength[i,3] = abs(dfMod_charLength[i,3]-lenVarDistanceEx)/lenVarDistanceEx
            dfMod_charLength[i,4] = abs(dfMod_charLength[i,4]-widVarDistanceEx)/widVarDistanceEx
            
        return dfMod_charLength
    
    #test function for one layer
    layer0 = charLengthCheck(0.2,0,dfEx,dfMod);
    dfDistMod.loc[:,'Path_charLengthW'] = layer0[:,2]
    dfDistMod.loc[:,'Path_charLengthSTDW'] = layer0[:,4]
    dfDistMod.loc[:,'Path_charLengthL'] = layer0[:,1]
    dfDistMod.loc[:,'Path_charLengthSTDL'] = layer0[:,3]
    
    #Plot Regular Coverage
    histRMS = dfDistMod.hist(column='coverage1', bins=10)
    #Plot ABC Coverage
    histXPercent = sortDistance(dfDistMod, 'Path_Coverage').head(numSaved).hist(column='coverage1', bins=10)
    
    convergenceCheck = sortDistance(dfDistMod, 'Path_Coverage');
    convergenceCounter = 0;
    for i in range(len(convergenceCheck.index)):
        if(convergenceCheck.loc[i,'Path_Coverage'] < error):
            convergenceCounter+=1;
        else:
            os.remove(path+'Generation/'+'model'+i)
            os.remove(path+'Scanning/'+'model'+i)
    if(convergenceCounter >= numConverged):
        convergence = True;
    else:
        convergence = False;
        
    #Plot Regular CharLength
    histRMS = dfDistMod.hist(column='charLengthL', bins=10)
    #Plot Sorted CharLength
    histXPercent = sortDistance(dfDistMod, 'Path_charLengthL').head(numSaved).hist(column='charLengthL', bins=10)