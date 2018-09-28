import dicom
import os
import numpy
from numpy.linalg import inv
import cv2
from Tkinter import *
import  Tkinter as Tk
import tkFileDialog
import nibabel as nib
from nibabel.processing import resample_from_to
from PIL import Image, ImageTk, ImageDraw
from skimage.measure import label
from skimage import filters
from skimage import measure
from skimage.morphology import skeletonize
import SimpleITK as sitk
import scipy
import csv
from random import randint
from copy import copy
import dcmstack
import time



#Global vars
lstRefFrameUID = [] #list of Ref Frame UIDs for each Exam
lstNifti = [] #list of Nifti files in the patient's folder
#lstExams = {} #dictionnary containing images, RF UID, type, etc. Index on lstRefFrameUID
RefSlice = 1
lstROIs = {} #dict of ROI with name and contours
TransfMatrix = {} #TransfMatrix(fix,moving) list of transformation matrix to an exam to an other
#W0 = 1200
#L0 = 615
flag_labeldone = 0 #set to 1 if labelisation is done
mouse_pos = [0,0]
t2seg_wind = [1190,905]#[W0, L0] #t2 segemntation window before binarization
patient = {} #dictionnary of patient informations
zoomF = 2 #zoom factor for image display


######Create main window
root = Tk.Tk()
root.title("MetaBrain - Yvan Pin")
root.geometry('{}x{}'.format(1200, 690))


##########################################
########Scroll Wheel management
def mouse_wheel(event):
    global ScrollWcount
    global RefSlice
    global vp_lab
    global vp_T1
    global vp_T2
    global vp_lab
    global datatoDisp_T1
    global datatoDisp_T2
    global datatoDisp_lab
    global flag_labeldone
    global zoomF
    
    # respond to Linux or Mac or Windows wheel event
    if event.num == 5 or event.delta == -1 or event.delta == -120:
        ScrollWcount -= 1
        RefSlice = RefSlice - 1
    if event.num == 4 or event.delta == 1 or event.delta == 120:
        ScrollWcount += 1
        RefSlice = RefSlice + 1

        ##########
    #for testing
 #   RefSlice = 136

    ###########
    res = datatoDisp_T1.shape
    #print RefSlice
    #####T1
    #img_matrix = Image.fromarray(datatoDisp_T1[:,:,RefSlice])#lstExams[selT1]['data'][str(RefSlice)][:,:])
    img_matrix = Image.fromarray(datatoDisp_T1[:,:,RefSlice,:], mode="RGB")
    img = ImageTk.PhotoImage(img_matrix)
 #   vp_T1 = Tk.Canvas(root, width = lstExams[selT1]['shape'][1], height = lstExams[selT1]['shape'][0])
    vp_T1.background = img
    #vp_T1.pack(side=TOP, anchor=NW)
    vp_T1.create_image(0,0, anchor = Tk.NW, image=img)
 #   vp_T1.place(x=0, y=0)


    #####FLAIR
    #img_matrix = Image.fromarray(datatoDisp_T1[:,:,RefSlice])#lstExams[selT1]['data'][str(RefSlice)][:,:])
    img_matrix = Image.fromarray(datatoDisp_T2[:,:,RefSlice,:], mode="RGB")
    img = ImageTk.PhotoImage(img_matrix.resize((res[1]*zoomF, res[0]*zoomF))) #   vp_T2 = Tk.Canvas(root, width = lstExams[selT2]['shape'][1], height = lstExams[selT2]['shape'][0])
    vp_T2.background = img
    #vp_T2.pack(side=TOP, anchor=NW)
    vp_T2.create_image(0,0, anchor = Tk.NW, image=img)
#    vp_T2.place(x=300, y=0)

    #####Label IMG
##    labimg_matrix = Image.fromarray(datatoDisp_lab[:,:,RefSlice])
##    labimg = ImageTk.PhotoImage(labimg_matrix)
##    vp_lab.background = labimg
##    vp_lab.create_image(0,0, anchor = Tk.NW, image=labimg)


##########################################
########Windowing segmentation
def t2windowing(event):
    global mouse_pos
    global t2seg_wind
    global datatoDisp_T2
    global dataRaw_T2
    global dataRGB_T2
    global RefSlice
    global vp_T2
    global zoomF
    #increment steps (px moved to produce new segementaion)
    istep = 10
    #print "(x,y) = ("+str(event.x)+","+str(event.y)+")"

    W2 = lstExams[selT2]['Wwidth']
    L2 = lstExams[selT2]['Wcenter']
    
    #if original position of mouse is 0,0, save the new pos
    if sum(mouse_pos) == 0:
        mouse_pos = [event.x, event.y]

    else:
        if (abs(mouse_pos[0]-event.x) == istep) or (abs(mouse_pos[1]-event.y) == istep):
            t2seg_wind = [t2seg_wind[0] + mouse_pos[0]-event.x, t2seg_wind[1] + mouse_pos[1]-event.y]
            #print t2seg_wind
            #update mouse pos saved
            mouse_pos = [event.x, event.y]

            #create a new segmentation representation of T2
            W = t2seg_wind[0]
            L = t2seg_wind[1]
            res = datatoDisp_T2.shape

            dataTemp = []
            dataTemp = copy(dataRaw_T2[:,:,RefSlice])
            dataTemp[dataTemp < (L-W/2)] = round((L-W/2))
            dataTemp[dataTemp > (L+W/2)] = round((L+W/2))
            dataTemp = dataTemp - round((L-W/2))
            print dataTemp.shape
            if numpy.sum(dataTemp) > 0:
                #gaussian filter on windowed image then threshold at 128 cv2.GaussianBlur(img,(5,5),0)
                dataTemp_bin = cv2.threshold(cv2.GaussianBlur(dataTemp,(5,5),0), 128, 255, cv2.THRESH_BINARY)[1]

                ####### Display limits of created egmentation onT2
                #create borders for display
                segmentedEdges = numpy.zeros(dataTemp.shape)
                edge_horizont = scipy.ndimage.sobel(dataTemp_bin, 0)
                edge_vertical = scipy.ndimage.sobel(dataTemp_bin, 1)
                #segmentedEdges = numpy.hypot(edge_horizont, edge_vertical)

                #segmntedEdges = scipy.ndimage.grey_erosion(segmentedEdges, size=(3,3))
                limits = numpy.hypot(edge_horizont, edge_vertical)
                limits[limits >1] = 1
                segmentedEdges = skeletonize(limits)
                #reset T2 FLAIR rhen add colors

                dataTempW = copy(dataRaw_T2[:,:,RefSlice])
                dataTempW[dataTempW < (L2-W2/2)] = round((L2-W2/2))
                dataTempW[dataTempW > (L2+W2/2)] = round((L2+W2/2))
                dataTempW = dataTempW - round((L2-W2/2))
                dataTempW = numpy.around((dataTempW*255)/W2)

                toDisp_T2 = numpy.empty((dataTemp.shape[0],dataTemp.shape[1],3), dtype=numpy.uint8)
                toDisp_T2[:,:,0] = copy(dataTempW)
                toDisp_T2[:,:,1] = copy(dataTempW)
                toDisp_T2[:,:,2] = copy(dataTempW)
                
                toDisp_T2[dataTemp_bin>0,2] = 160
##                toDisp_T2[segmentedEdges>0,0] = 150
##                toDisp_T2[segmentedEdges>0,1] = 0
##                toDisp_T2[segmentedEdges>0,2] = 0
                print toDisp_T2.shape
                img_matrix = Image.fromarray(toDisp_T2, mode="RGB")
                img = ImageTk.PhotoImage(img_matrix.resize((res[1]*zoomF, res[0]*zoomF))) #   vp_T2 = Tk.Canvas(root, width = lstExams[selT2]['shape'][1], height = lstExams[selT2]['shape'][0])
                vp_T2.background = img
                vp_T2.create_image(0,0, anchor = Tk.NW, image=img)
##                #display the labelImg (8bits pixels Black and White), adjuted depending the max value (just for display)

##                labimg_matrix = Image.fromarray(dataTemp_bin)
##                labimg = ImageTk.PhotoImage(labimg_matrix)
##                vp_lab.background = labimg
##                vp_lab.create_image(0,0, anchor = Tk.NW, image=labimg)


                        
            #find contours, not working
##            edge_horizont = scipy.ndimage.sobel(dataTemp_bin, 0)
##            edge_vertical = scipy.ndimage.sobel(dataTemp_bin, 1)
##            segmentedEdges = numpy.hypot(edge_horizont, edge_vertical)
##            datatoDisp_T2 = dataRaw_T2
##            datatoDisp_T2[segmentedEdges>0,RefSlice,1] = 180


############################################
######## Reset mouse position after segmentation windowing when button release
def mouse_release(event):
    global mouse_pos
    global dataRaw_T2
    global t2seg_wind
    global T2seg_mask
    global datatoDisp_lab
    global datatoDisp_T2
    global dataRGB_T2
    global dataRawRGB_T2
    global flag_labeldone
    global selT1
    global selT2
    #FLAIR windowing

    W2 = lstExams[selT2]['Wwidth']
    L2 = lstExams[selT2]['Wcenter'] 

    if flag_labeldone == 0:
    
        mouse_pos =  [0, 0]
        T2seg_mask = numpy.zeros(dataRaw_T2.shape)

        #create a binary mask in 3D of the T2 based on the window selected by mouse move+B1
        Ws = t2seg_wind[0]
        Ls = t2seg_wind[1]
        print "L/W = " + str(Ls) + "/" + str(Ws)

        dataTemp = copy(dataRaw_T2)
        dataTemp[dataTemp < (Ls-Ws/2)] = round((Ls-Ws/2))
        dataTemp[dataTemp > (Ls+Ws/2)] = round((Ls+Ws/2))
        dataTemp = dataTemp - round((Ls-Ws/2))
        
        dataTempW = copy(dataRaw_T2)
        dataTempW[dataTempW < (L2-W2/2)] = round((L2-W2/2))
        dataTempW[dataTempW > (L2+W2/2)] = round((L2+W2/2))
        dataTempW = dataTempW - round((L2-W2/2))
        dataWind_T2 = numpy.around((dataTempW*255)/W2)


        for s in range(0,dataTemp.shape[2]):
            #if numpy.sum(dataTemp) > 0:
                    #gaussian filter on windowed image then threshold at 128 cv2.GaussianBlur(img,(5,5),0)
            T2seg_mask[:,:,s] = cv2.threshold(cv2.GaussianBlur(dataTemp[:,:,s],(5,5),0), 128, 255, cv2.THRESH_BINARY)[1]

                
        #datatoDisp_lab = T2seg_mask*255

        ####### Display limits of created egmentation onT2
        #create borders for display
##        segmentedEdges = []
##        segmentedEdges = numpy.zeros(T2seg_mask.shape)
##        for s in range(T2seg_mask.shape[2]-1):
##            edge_horizont = scipy.ndimage.sobel(T2seg_mask[:,:,s], 0)
##            edge_vertical = scipy.ndimage.sobel(T2seg_mask[:,:,s], 1)
##            limits = numpy.hypot(edge_horizont, edge_vertical)
##            limits[limits >1] = 1
##            segmentedEdges[:,:,s] = skeletonize(limits)

        #reset T2 FLAIR then add colors
        datatoDisp_T2[:,:,:,:] = 0
        datatoDisp_T2[:,:,:,0] = dataWind_T2[:,:,:]
        datatoDisp_T2[:,:,:,1] = dataWind_T2[:,:,:]
        datatoDisp_T2[:,:,:,2] = dataWind_T2[:,:,:]

        datatoDisp_T2[T2seg_mask>0,2] = 160
        #datatoDisp_T2[segmentedEdges>0,0] = 100
        #datatoDisp_T2[segmentedEdges>0,1] = 0
        #datatoDisp_T2[segmentedEdges>0,2] = 0      
        #datatoDisp_T2[segmentedEdges>0,2] = 255
    
##########################################    
########GET PATIENT'S DIR (after selection)
def getPatientDir():
    #global lstNifti
    global lstExams
    patientDir = tkFileDialog.askdirectory()
    print "Patient's directory selected : " + patientDir
    #lauch dicom reading
    # lstExams,lstRefFrameUID = readDicoms(patientDir, 'MR')
    lstExams = readDicoms(patientDir, 'MR')
    #print lstExams
    #display exams in T1 MPR and FLAIR selection list
    #for key,exam in lstExams.iteritems():
    for iexam in range(0, len(lstExams), 1):
##        lstbxT1.insert('end',exam['description'][exam['description'].find('; ')+2:len(exam['description'])])
##        lstbxT2.insert('end',exam['description'][exam['description'].find('; ')+2:len(exam['description'])])
##        lstbxT1.insert('end', os.path.basename(examfile))
##        lstbxT2.insert('end',os.path.basename(examfile))
        lstbxT1.insert('end', lstExams[iexam]['description'])
        lstbxT2.insert('end', lstExams[iexam]['description'])
##########################################
########FIND MRI ID IN LOG FILE###########
def findMRI_ID(log_str):
   "findMRI_ID function launched..."
   lstMRI_ID = []
   return [lstMRI_ID]


##########################################
########FIND DICOMS with a specified type (ex 'MR')###########
def readDicoms(p_dir, dcm_type):
    #global lstExams
    global lstROIs
    global patient
    print "> readDicoms launched with type " + dcm_type
    #init
    lstExams = {}
    lstFilesDCM = []  # create an empty list for DCM images
    lstFilesDCMsorted = {}  #empty list of files, after sorting by study
 #  lstFilesNIFTI = [] # create an empty list for NIFTI

    #list the dcms/nifti
    for dirName, subdirList, fileList in os.walk(p_dir):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
 #           elif ".nii.gz" in filename.lower():  # check whether the file's NIFTI
 #               lstFilesNIFTI.append(os.path.join(dirName,filename))
    #print lstFilesDCM


    #### Find the Structure Set.
    for filenameDCM in lstFilesDCM:
        if "rts" in filenameDCM.lower():  # check whether the file's RTStruct
            # read the file
            ds = dicom.read_file(filenameDCM)

            ####Get patient's name, surname, birthdate, NIP
            patient['name'] = ds.PatientsName
            patient['id'] = ds.PatientID
            patient['birthdate'] = ds.PatientsBirthDate
            ####Get contours
            print '### GET CONTOURS : ###'
            ##ROI Names
            for ROI in ds.StructureSetROIs:
                lstROIs[int(ROI.ROINumber)] = {}
                lstROIs[int(ROI.ROINumber)]['name'] = ROI.ROIName
                lstROIs[int(ROI.ROINumber)]['rfUID'] = ROI.ReferencedFrameofReferenceUID
                lstROIs[int(ROI.ROINumber)]['color'] = ds.ROIContourSequence[int(ROI.ROINumber)].ROIDisplayColor
                #print "Color = " + str(lstROIs[int(ROI.ROINumber)]['color'])
                lstROIs[int(ROI.ROINumber)]['contours'] = numpy.empty([3,1])
                #get contours
                for sliceSeq in ds.ROIContours[int(ROI.ROINumber)].Contours:
                    rSOPUID = sliceSeq.ContourImages[0].ReferencedSOPInstanceUID
                    #print rSOPUID
                    contData = sliceSeq.ContourData
                    #save each point as a xyz triplet
                    x = -numpy.array([contData[::3]])
                    y = -numpy.array([contData[1::3]])
                    z = numpy.array([contData[2::3]])
                    #changing base matrix preparation : to go from mm to pixels number, see DICOM STANDARD Equation C.7.6.2.1-1.
                    P = numpy.vstack((x,y,z))  #slice matrix #Pxyz
##                    S = lstExams[examIndex]['data'][str(instance)]['ImagePosition']
##                    changebaseMatrix = numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                    coords = numpy.vstack((x,y,z))  #slice matrix
                    
                    lstROIs[int(ROI.ROINumber)]['contours'] = numpy.concatenate((lstROIs[int(ROI.ROINumber)]['contours'], coords), axis=1)  #save points coords (x,y,z)
                print "--- " + ROI.ROIName + " : " + str(lstROIs[int(ROI.ROINumber)]['contours'].shape)

            ####Get tranformation matrixs
            
            for RFOR in ds.ReferencedFrameofReferences:
                if "FrameofReferenceRelationships" in RFOR :    #If FofRR exists : get the transformation matrix
                    print "RFR exists"
                
                    toUID = RFOR.FrameofReferenceUID #exam non moving
                    #find internal exam ID from UID
                    toID = int(lstRefFrameUID.index(toUID))
                    #Initialise a 2D matrix
                    TransfMatrix[toID] = {}
                    #save an identity matrix if fromID would be the same that toID
                    TransfMatrix[toID][toID] = numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                    #for the non moving exam, find all moved exam and homogeneous matrix corresponding for registration
                    for fromRFOR in RFOR.FrameofReferenceRelationships:
                        fromUID = fromRFOR.RelatedFrameofReferenceUID #exam moving
                        #find internal exam ID from UID
                        fromID = int(lstRefFrameUID.index(fromUID))
                        #format the homogeneous matrix to be 4*4
                        MHomogeneous = numpy.matrix([fromRFOR.FrameofReferenceTransformationMatrix[:4],fromRFOR.FrameofReferenceTransformationMatrix[4:8],fromRFOR.FrameofReferenceTransformationMatrix[8:12],fromRFOR.FrameofReferenceTransformationMatrix[12:16]])
                        #print str(fromID) +" to "+ str(toID) + " : "
                        #print fromRFOR.FrameofReferenceTransformationComment
                        #save it
                        TransfMatrix[toID][fromID] = MHomogeneous
                        #print TransfMatrix[toID][fromID]

                else:
                    print "RFR doesn't exists"  #no transformation known in dicom, then generate an identity matrix

                    #create a standard identity matrix for five scans (hypothetic)
                    identityM = numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                    
                    for toID in range(1,5) :
                        TransfMatrix[toID] = {}
                        for fromID in range(1,5) :
                            TransfMatrix[int(toID)][int(fromID)] = identityM
                            
                    print TransfMatrix
            
        else:
            # read the file
            ds = dicom.read_file(filenameDCM)
            
            #reference frame UID
            rfUID = ds[0x20,0x52].value
            #instance nb
            instance = int(ds[0x20,0x13].value)
            #print instance

            if rfUID in lstRefFrameUID:     #rfUID already known
                examIndex = int(lstRefFrameUID.index(rfUID))
            else:
                lstRefFrameUID.append(rfUID)        #rfUID unknown -> add to the list
                examIndex = int(lstRefFrameUID.index(rfUID))
                print str(examIndex)+' - '+str(instance)
                #print "New index" + str(examIndex)
                #save modality, exam descr
                lstExams[examIndex] = {}
                lstExams[examIndex]['modality'] = ds[0x08,0x60].value
                lstExams[examIndex]['description'] = ds[0x08,0x103e].value
##                lstExams[examIndex]['rows'] = int(ds.Rows)
##                lstExams[examIndex]['columns'] = int(ds.Columns)
                #lstExams[examIndex]['PixelSpacing'] = ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
                lstExams[examIndex]['Wcenter'] = ds.WindowCenter
                lstExams[examIndex]['Wwidth'] = ds.WindowWidth
                #lstExams[examIndex]['origin'] = 0
                
                #create data list for saving positions data
                #lstExams[examIndex]['positions'] = {}
                #create the data list for saving voxels
##                lstExams[examIndex]['data'] = {}
##                lstExams[examIndex]['data0'] = {}
##                lstExams[examIndex]['affine'] = numpy.zeros((4,4))

                #save file path after sorting
            #print filenameDCM
            if len(lstFilesDCMsorted) == examIndex:
                lstFilesDCMsorted[examIndex] = {}
            lstFilesDCMsorted[examIndex][len(lstFilesDCMsorted[examIndex])] = filenameDCM
 
            #print "instance from " + str(examIndex) + " = " + str(instance)
            #save pixels in array
            #create a list for all specific data from the slice
##            lstExams[examIndex]['data0'][str(instance)] = {}
##            lstExams[examIndex]['data0'][str(instance)]['pixels'] = numpy.zeros((int(ds.Rows), int(ds.Columns)), dtype=ds.pixel_array.dtype) #numpy.zeros((int(ds.Rows), int(ds.Columns)))
##            lstExams[examIndex]['data0'][str(instance)]['pixels'] = ds.pixel_array
##            lstExams[examIndex]['data0'][str(instance)]['SOPInstanceUID'] = str(ds.SOPInstanceUID)
##            lstExams[examIndex]['data0'][str(instance)]['ImagePosition'] = ds.ImagePositionPatient
##            lstExams[examIndex]['data0'][str(instance)]['ImageOrientation'] = ds.ImageOrientationPatient
 


    
    ####Create anifti object for each exam and extract all nifti data
    print "--> DCMSTACK"
    for iexam in range(0, len(lstFilesDCMsorted), 1):
        src_paths = list(lstFilesDCMsorted[iexam].values())
        my_stack = dcmstack.DicomStack()
        for src_path in src_paths:
            src_dcm = dicom.read_file(src_path)
            my_stack.add_dcm(src_dcm)

        #create nifti
        niimg = my_stack.to_nifti()

        #save modality, exam descr
        #lstExams[examIndex] = {}
        #lstExams[examIndex]['modality'] = 'MR'
        #lstExams[examIndex]['description'] = os.path.basename(filenameNIFTI)
        lstExams[iexam]['shape'] = niimg.shape
        lstExams[iexam]['PixelSpacing'] = niimg.header['pixdim']
        lstExams[iexam]['affine'] = niimg.affine
        #lstExams[examIndex]['Wcenter'] = ds.WindowCenter
        #lstExams[examIndex]['Wwidth'] = ds.WindowWidth
        #lstExams[examIndex]['origin'] = 0
        #create data list for saving positions data
        #lstExams[examIndex]['positions'] = {}
        #create the data list for saving voxels
        lstExams[iexam]['data'] = niimg.get_data()
        lstExams[iexam]['raw'] = niimg


    
 #   print stacks
##        examM = numpy.zeros((lstExams[iexam]['data0']['0']['pixels'].shape[0], lstExams[iexam]['data0']['0']['pixels'].shape[1], len(lstExams[iexam]['data0'])), dtype=numpy.uint8)
##        for islice in range(0, len(lstExams[iexam]['data0']), 1):
##            examM[:,:,islice] = lstExams[iexam]['data0'][str(islice)]['pixels']
##        #save the 3D matrix in ['data']
##        print examM.shape
##        lstExams[iexam]['data'] = examM
##        #save shape
##        lstExams[iexam]['shape'] = {}
##        lstExams[iexam]['shape'][0] = lstExams[iexam]['data0']['0']['pixels'].shape[0]
##        lstExams[iexam]['shape'][1] = lstExams[iexam]['data0']['0']['pixels'].shape[1]
##        lstExams[iexam]['shape'][2] = len(lstExams[iexam]['data0'])




        #### Read the Nifti images
##        examIndex = 0
##    for filenameNIFTI in lstFilesNIFTI:
##        niimg = nib.load(filenameNIFTI)
##        hdr = niimg.header
##        
##        #save modality, exam descr
##        lstExams[examIndex] = {}
##        lstExams[examIndex]['modality'] = 'MR'
##        lstExams[examIndex]['description'] = os.path.basename(filenameNIFTI)
##        lstExams[examIndex]['shape'] = niimg.shape
##        lstExams[examIndex]['PixelSpacing'] = hdr['pixdim']
##        lstExams[examIndex]['affine'] = niimg.affine
##        #lstExams[examIndex]['Wcenter'] = ds.WindowCenter
##        #lstExams[examIndex]['Wwidth'] = ds.WindowWidth
##        #lstExams[examIndex]['origin'] = 0
##        #create data list for saving positions data
##        #lstExams[examIndex]['positions'] = {}
##        #create the data list for saving voxels
##        lstExams[examIndex]['data'] = niimg.get_data()
##        lstExams[examIndex]['raw'] = niimg
##        print lstExams[examIndex]['data'].shape
##        examIndex = examIndex+1



            
            ####Get tranformation matrixs
##            for RFOR in ds.ReferencedFrameofReferences:
##                toUID = RFOR.FrameofReferenceUID #exam non moving
##                #find internal exam ID from UID
##                toID = int(lstRefFrameUID.index(toUID))
##                #Initialise a 2D matrix
##                TransfMatrix[toID] = {}
##                #save an identity matrix if fromID would be the same that toID
##                TransfMatrix[toID][toID] = numpy.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
##                #for the non moving exam, find all moved exam and homogeneous matrix corresponding for registration
##                for fromRFOR in RFOR.FrameofReferenceRelationships:
##                    fromUID = fromRFOR.RelatedFrameofReferenceUID #exam moving
##                    #find internal exam ID from UID
##                    fromID = int(lstRefFrameUID.index(fromUID))
##                    #format the homogeneous matrix to be 4*4
##                    MHomogeneous = numpy.matrix([fromRFOR.FrameofReferenceTransformationMatrix[:4],fromRFOR.FrameofReferenceTransformationMatrix[4:8],fromRFOR.FrameofReferenceTransformationMatrix[8:12],fromRFOR.FrameofReferenceTransformationMatrix[12:16]])
##                    #print str(fromID) +" to "+ str(toID) + " : "
##                    #print fromRFOR.FrameofReferenceTransformationComment
##                    #save it
##                    TransfMatrix[toID][fromID] = MHomogeneous
##                    #print TransfMatrix[toID][fromID]
##
##                print TransfMatrix
            
##        else:
##            # read the file
##            ds = dicom.read_file(filenameDCM)
##            
##            #reference frame UID
##            rfUID = ds[0x20,0x52].value
##            #instance nb
##            instance = int(ds[0x20,0x13].value)
##            #print instance
##
##            if rfUID in lstRefFrameUID:     #rfUID already known
##                examIndex = int(lstRefFrameUID.index(rfUID))
##            else:
##                lstRefFrameUID.append(rfUID)        #rfUID unknown -> add to the list
##                examIndex = int(lstRefFrameUID.index(rfUID))
##                #print "New index" + str(examIndex)
##                #save modality, exam descr
##                lstExams[examIndex] = {}
##                lstExams[examIndex]['modality'] = ds[0x08,0x60].value
##                lstExams[examIndex]['description'] = ds[0x08,0x103e].value
##                lstExams[examIndex]['rows'] = int(ds.Rows)
##                lstExams[examIndex]['columns'] = int(ds.Columns)
##                lstExams[examIndex]['PixelSpacing'] = ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
##                lstExams[examIndex]['Wcenter'] = ds.WindowCenter
##                lstExams[examIndex]['Wwidth'] = ds.WindowWidth
##                lstExams[examIndex]['origin'] = 0
##                #create data list for saving positions data
##                #lstExams[examIndex]['positions'] = {}
##                #create the data list for saving voxels
##                lstExams[examIndex]['data'] = {}
## 
##
##            #save pixels in array
##            #create a list for all specific data from the slice
##            lstExams[examIndex]['data'][str(instance)] = {}
##            lstExams[examIndex]['data'][str(instance)]['pixels'] = numpy.zeros((320, 240), dtype=ds.pixel_array.dtype) #numpy.zeros((int(ds.Rows), int(ds.Columns)))
##            lstExams[examIndex]['data'][str(instance)]['pixels'] = ds.pixel_array
##            lstExams[examIndex]['data'][str(instance)]['SOPInstanceUID'] = str(ds.SOPInstanceUID)
##            lstExams[examIndex]['data'][str(instance)]['ImagePosition'] = ds.ImagePositionPatient
##            lstExams[examIndex]['data'][str(instance)]['ImageOrientation'] = ds.ImageOrientationPatient
##            
##            #if instance == 0
## #              lstExams[examIndex]['origin'] = 
##
###            lstExams[examIndex]['positions'][str(instance)] = {}
###            lstExams[examIndex]['positions'][str(instance)]['']
            
    #print lstRefFrameUID
    print "###" + str(len(lstExams)) + " exam(s) found ###"
    #lstNifti = dict(zip(range(0,len(lstFilesNIFTI)-1), lstFilesNIFTI[0::1]))
    return lstExams #,lstRefFrameUID


##########################################
####Handle listbox seletion and exam display ###################
###########T1
def displayT1(event):
    global vp_T1
    global vp_lab
    global selT1
    global labimg
    global labelImg #contains all contours encoded by ROIid+1
    global datatoDisp_lab #labelImg encoded between 0-255 to display only
    global dataRaw_T1
    global dataRGB_T1
    global datatoDisp_T1
    global RefSlice
    global lstNifti
    #global W0
    #global L0
    global lstExams
    global zoomF

    
    print "> Launching displayT1 for " + lstbxT1.get(lstbxT1.curselection())
    selT1 = lstbxT1.curselection()[0]

    #T1 windowing
    W1 = lstExams[selT1]['Wwidth']
    L1 = lstExams[selT1]['Wcenter']
    
    print selT1
    RefSlice = 0
    #create a Grayscale 3D matrix from image
    for nslice in range(lstExams[selT1]['shape'][2]-1):
       #print  nslice
       if nslice == 0:
            img3D = lstExams[selT1]['data'][:,:,nslice]
 #                    img3D = lstExams[selT1]['data'][0::lstExams[selT1]['shape'][0]-1][0::lstExams[selT1]['shape'][1]-1][nslice]
       else:
            img3D = numpy.dstack((img3D,lstExams[selT1]['data'][:,:,nslice]))
    dataRaw_T1 = img3D
    print "--- Created a 3D image data for T1 (exam " + str(selT1) + ") :"

    # get image data from Nifti file
    #dataRaw_T1 = nib.load(lstNifti[selT1])
    
    #print dataRaw_T1.shape
    #image windowing
    #dataTemp = dataRaw_T1 - (L0-W0/2)
    dataTemp = dataRaw_T1
    #print numpy.amin(dataTemp)
    #print numpy.amax(dataTemp)
    dataTemp[dataRaw_T1 < (L1-W1/2)] = round((L1-W1/2))
    dataTemp[dataRaw_T1 > (L1+W1/2)] = round((L1+W1/2))
    dataTemp = dataTemp - round((L1-W1/2))
    dataWind_T1 = numpy.around((dataTemp*255)/W1)
    #print numpy.amin(dataTemp)
    #print numpy.amax(dataTemp)

    #create an empty matrix of MPR's shape for labelisation
    labelImg = numpy.zeros(dataRaw_T1.shape, dtype=numpy.uint8)
    
    #create an RGB matrix for countour display
    dataRGB_T1 = numpy.empty((lstExams[selT1]['shape'][0], lstExams[selT1]['shape'][1],  lstExams[selT1]['shape'][2]-1,3), dtype=numpy.uint8)
        #first insert the grayscale image (encoded in RGB)
    dataRGB_T1[:, :, :, 0] = dataWind_T1
    dataRGB_T1[:, :, :, 1] = dataWind_T1
    dataRGB_T1[:, :, :, 2] = dataWind_T1

        ###then color the contours
    for ROIid in lstROIs:
        #transformation from coords in millimeters to MPR's voxel (inverse affine matrix)
        vcoords = numpy.rint(nib.affines.apply_affine(inv(lstExams[selT1]['affine']),lstROIs[ROIid]['contours'].transpose())).astype(int).transpose()
        vcoords[vcoords<0] = 0 #security
##        dataRGB_T1[int(lstROIs[ROIid]['contours'][0]), int(lstROIs[ROIid]['contours'][1]), int(lstROIs[ROIid]['contours'][2]), 0] = int(lstROIs[ROIid]['color'][0])
##        dataRGB_T1[int(lstROIs[ROIid]['contours'][0]), int(lstROIs[ROIid]['contours'][1]), int(lstROIs[ROIid]['contours'][2]), 1] = int(lstROIs[ROIid]['color'][1])
##        dataRGB_T1[int(lstROIs[ROIid]['contours'][0]), int(lstROIs[ROIid]['contours'][1]), int(lstROIs[ROIid]['contours'][2]), 2] = int(lstROIs[ROIid]['color'][2])
        dataRGB_T1[vcoords[0], vcoords[1], vcoords[2], 0] = int(lstROIs[ROIid]['color'][0])
        dataRGB_T1[vcoords[0], vcoords[1], vcoords[2], 1] = int(lstROIs[ROIid]['color'][1])
        dataRGB_T1[vcoords[0], vcoords[1], vcoords[2], 2] = int(lstROIs[ROIid]['color'][2])

        #fill the labelisation image with contour points, gray level correspond to the ROIid+1, so all contours are identificable on one image
        #print "ROIid = " + str(ROIid)
        labelImg[vcoords[0], vcoords[1], vcoords[2]] = ROIid+1

    datatoDisp_T1 = dataRGB_T1
    #add ROIs
##    for ROI in lstROIs:
##        fromID = int(lstRefFrameUID.index([ROI['rfUID']]))
##        coordsT = numpy.dot(ROI['contours'],TransfMatrix[selT1][fromID])
##        #round coords to fit exam slices
##        datatoDisp_T1
    #create the viewport
    #print lstExams[selT1]['data']
    img_matrix = Image.fromarray(datatoDisp_T1[:,:,RefSlice,:], mode="RGB")#lstExams[selT1]['data'][str(RefSlice)][:,:])
    img = ImageTk.PhotoImage(img_matrix)
    vp_T1 = Tk.Canvas(root, width = lstExams[selT1]['shape'][1], height = lstExams[selT1]['shape'][0])
    vp_T1.background = img
    #vp_T1.pack(side=TOP, anchor=NW)
    vp_T1.create_image(0,0, anchor = Tk.NW, image=img)
    vp_T1.place(x=0, y=0)


    #display the labelImg (8bits pixels Black and White), adjuted depending the max value (just for display)
    datatoDisp_lab = numpy.round(labelImg*(255/numpy.amax(labelImg)))
##    labimg_matrix = Image.fromarray(datatoDisp_lab[:,:,RefSlice])#lstExams[selT1]['data'][str(RefSlice)][:,:])
##    labimg = ImageTk.PhotoImage(labimg_matrix)
##    vp_lab = Tk.Canvas(root, width = lstExams[selT1]['shape'][1], height = lstExams[selT1]['shape'][0])
##    vp_lab.background = labimg
##    vp_lab.create_image(0,0, anchor = Tk.NW, image=labimg)
##    vp_lab.place(x=600, y=0)
###########T2
def displayT2(event):
    global vp_T2
    global selT2
    global datatoDisp_T2
    global dataRaw_T2
    global dataRawRGB_T2
    global dataRGB_T2
    global zoomF
    
    print "> Launching displayT2 for " + lstbxT2.get(lstbxT2.curselection())
    selT2 = lstbxT2.curselection()[0]

    #FLAIR windowing
    W2 = lstExams[selT2]['Wwidth']
    L2 = lstExams[selT2]['Wcenter'] 

    #Sampling the FLAIE on MPR
    print "Sampling FLAIR " +str(selT2)+ " on T1 MPR " + str(selT1)
    RefSlice = 0
    print "Initial shape is " + str(lstExams[selT2]['raw'].get_data().shape)
    lstExams[selT2]['raw'] = resample_from_to(lstExams[selT2]['raw'],lstExams[selT1]['raw'])
    print "Sampled shape is " +str(lstExams[selT2]['raw'].get_data().shape)

    #Updating data and shape values after sampling
    lstExams[selT2]['data'] = lstExams[selT2]['raw'].get_data()
    lstExams[selT2]['shape'] = lstExams[selT2]['raw'].get_data().shape

    #create a Grayscale 3D matrix from image
    for nslice in range(lstExams[selT2]['shape'][2]-1):
       #print  nslice
       if nslice == 0:
            img3D = lstExams[selT2]['data'][:,:,nslice]
 #                    img3D = lstExams[selT2]['data'][0::lstExams[selT2]['shape'][0]-1][0::lstExams[selT2]['shape'][1]-1][nslice]
       else:
            img3D = numpy.dstack((img3D,lstExams[selT2]['data'][:,:,nslice]))
    dataRaw_T2 = img3D

    print "--- Created a 3D image data for T2 FLAIR (exam " + str(selT2) + ") :"

    # get image data from Nifti file
    #dataRaw_T1 = nib.load(lstNifti[selT1])

    #image windowing
    dataTemp = dataRaw_T2[:,:,:]
    dataTemp[dataRaw_T2 < (L2-W2/2)] = round((L2-W2/2))
    dataTemp[dataRaw_T2 > (L2+W2/2)] = round((L2+W2/2))
    dataTemp = dataTemp - round((L2-W2/2))
    dataWind_T2 = numpy.around((dataTemp*255)/W2)
    #create an RGB matrix for countour display
    dataRGB_T2 = numpy.empty((lstExams[selT2]['shape'][0], lstExams[selT2]['shape'][1],  lstExams[selT2]['shape'][2]-1,3), dtype=numpy.uint8)
    dataRGB_T2[:, :, :, 0] = dataWind_T2[:,:,:]
    dataRGB_T2[:, :, :, 1] = dataWind_T2[:,:,:]
    dataRGB_T2[:, :, :, 2] = dataWind_T2[:,:,:]#dataWind_T1
    datatoDisp_T2 = dataRGB_T2[:,:,:,:]

    #create a data raw in RGD
##    dataRawRGB_T2 = numpy.empty((lstExams[selT2]['shape'][0], lstExams[selT2]['shape'][1],  lstExams[selT2]['shape'][2]-1,3), dtype=numpy.uint8)
##    dataRawRGB_T2 = datatoDisp_T2        
##    dataRawRGB_T2[:,:,:,0] = dataRaw_T2
##    dataRawRGB_T2[:,:,:,1] = dataRaw_T2
##    dataRawRGB_T2[:,:,:,2] = dataRaw_T2

    
    #create the viewport
    #print selT2
    #print lstExams[selT2]['data']['1']
    img_matrix = Image.fromarray(datatoDisp_T2[:,:,RefSlice,:], mode="RGB")#lstExams[selT1]['data'][str(RefSlice)][:,:])
    img = ImageTk.PhotoImage(img_matrix)
    vp_T2 = Tk.Canvas(root, width = lstExams[selT2]['shape'][1]*zoomF, height = lstExams[selT2]['shape'][0]*zoomF)
    vp_T2.background = img
    vp_T2.create_image(0,0, anchor = Tk.NW, image=img)
    vp_T2.place(x=300, y=0)





###########Segmentation function
def segment_flair():
    global lstROIs
    global lstExams
    global selT1
    global dataRaw_T1
    global labelImg
    global flag_labeldone
    global datatoDisp_lab
    global datatoDisp_T2
    count_gtvs = 0

    print "Launching segmentation..."

    #reset T2 display
    W2 = lstExams[selT2]['Wwidth']
    L2 = lstExams[selT2]['Wcenter']
    dataTempW = copy(dataRaw_T2)
    dataTempW[dataTempW < (L2-W2/2)] = round((L2-W2/2))
    dataTempW[dataTempW > (L2+W2/2)] = round((L2+W2/2))
    dataTempW = dataTempW - round((L2-W2/2))
    dataTempW = numpy.around((dataTempW*255)/W2)

    datatoDisp_T2[:,:,:,0] = copy(dataTempW)
    datatoDisp_T2[:,:,:,1] = copy(dataTempW)
    datatoDisp_T2[:,:,:,2] = copy(dataTempW)

    #define closing kernel
    kernelC = numpy.ones((15,15),numpy.uint8)
    kernelD = numpy.ones((5,5),numpy.uint8)
    
    #reset global datatoDisp_lab
    datatoDisp_lab[:,:,:] = 0
    for ROIid in lstROIs:
        tosegmentM = T2seg_mask #T2seg_mask = edema binary image of all MRI exam
        #foreach ROIs, if found 1 call "gtv...", continue the segmentation process
        if "gtv" in lstROIs[ROIid]['name'].lower(): # and count_gtvs == 0:
            print "--- Segment GTV '" +lstROIs[ROIid]['name']+ "'"
            #(1) create a zeros working Matrix "labelM" for masking and an other matrix for seglentation
            labelM = numpy.zeros(dataRaw_T1.shape, dtype=numpy.uint8)  #shape of the T1 reference data
 
            #(1.2) set to 1 labelM pixels corresponding in countours (saved in labelImg, with intensity ROIid+1)
            labelM[labelImg==ROIid+1] = 1
            #(2) polygone mask generation for each slice
            for nslice in range(labelM.shape[2]-1):
                if numpy.sum(labelM[:,:,nslice]) > 0:   #if  a contour is on the slice
                    #apply affine matrix on raw coords
                    af_coords = nib.affines.apply_affine(inv(lstExams[selT1]['affine']),lstROIs[ROIid]['contours'].transpose())
                    vcoords = numpy.rint(af_coords).astype(int).transpose()
                    #print vcoords.shape
                    vcoords[vcoords<0] = 0 #security
                    
                    #points on slice (index)
                    sl_cont_indexs = [index for index, value in enumerate(vcoords[2]) if value == nslice] #int(numpy.where(vcoords[2]==nslice)[0])
                    #print sl_cont_indexs

                    vcoordsX = []
                    vcoordsY = []

                    for pt_index in sl_cont_indexs:
                        vcoordsX.append(vcoords[1, pt_index])
                        vcoordsY.append(vcoords[0, pt_index])
                    
                    #create tuples of x;y coords
                    polygon = zip(vcoordsX, vcoordsY)
                    #print polygon
                    #draw polygone
                    img = Image.new('L', (labelM.shape[1], labelM.shape[0]), 0)
                    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                    mask = numpy.array(img)
                    
                    labelM[:,:,nslice] = mask

                    
                    
            

            #save GTV segmentation
            lstROIs[ROIid]['GTVseg'] = labelM
            #fuse the tosegmentM with labelM to fill holes of metastasis in tosegmentM
            tosegmentM = tosegmentM + labelM
            tosegmentM[tosegmentM>0] = 1
            #datatoDisp_lab[tosegmentM>0] = 255

            ### region growing segmentaiton
                #convert numpy to sitk image
            tosegmentM_sitk = sitk.GetImageFromArray(tosegmentM)
                # SEED
            #ROIpts_coords = numpy.rint(nib.affines.apply_affine(inv(lstExams[selT1]['affine']),lstROIs[ROIid]['contours'].transpose())).astype(int).transpose()
            ROIpts_coords = nib.affines.apply_affine(inv(lstExams[selT1]['affine']),lstROIs[ROIid]['contours'].transpose()).transpose()
            #seed = (int(numpy.mean(lstROIs[ROIid]['contours'][0])),int(numpy.mean(lstROIs[ROIid]['contours'][1])),int(numpy.mean(lstROIs[ROIid]['contours'][2])))
            seed = (int(numpy.rint(numpy.mean(ROIpts_coords[0]))),int(numpy.rint(numpy.mean(ROIpts_coords[1]))),int(numpy.rint(numpy.mean(ROIpts_coords[2]))))
            #print seed
            #enlarge seed
            

            segmentedM, nf =  scipy.ndimage.label(tosegmentM)
            if nf > 0:
                segmentedM[segmentedM != segmentedM[seed]] = 0

            #save EDEMA segmentation
            lstROIs[ROIid]['EDEMAseg'] = segmentedM

            #calcul volume of T1 voxel in mm3, GTV and Edema in Voxel
                #security
            labelM[:,:,-dataRaw_T1.shape[2]:0] = 0
            segmentedM[:,:,-dataRaw_T1.shape[2]:0] = 0
            T1vxldim = lstExams[selT1]['PixelSpacing'] #1,2,3
            lstROIs[ROIid]['VxlVol'] = T1vxldim[1]*T1vxldim[3]*T1vxldim[3] #in mm3
            lstROIs[ROIid]['GTVvol'] = numpy.count_nonzero(labelM)    #in voxels
            lstROIs[ROIid]['EDEMAvol'] = numpy.count_nonzero(segmentedM) #in voxels

            print "---- GTV = " +str(lstROIs[ROIid]['GTVvol'])+ " vxl, EDEMA = " +str(lstROIs[ROIid]['EDEMAvol'])+ " vxl, EDEMA/GTV = " +str(lstROIs[ROIid]['EDEMAvol']/float(lstROIs[ROIid]['GTVvol']))+ "; (vxl = " +str(lstROIs[ROIid]['VxlVol'])+ " mm3)"

            
            ###display#######                    
            #create borders for display
            segmentedEdges = numpy.zeros(segmentedM.shape)
            for s in range(segmentedM.shape[2]-1):
                edge_horizont = scipy.ndimage.sobel(segmentedM[:,:,s], 0)
                edge_vertical = scipy.ndimage.sobel(segmentedM[:,:,s], 1)
                limits = numpy.hypot(edge_horizont, edge_vertical)
                limits[limits >1] = 1
                segmentedEdges[:,:,s] = skeletonize(limits)
                #segmentedEdges[:,:,s] = numpy.hypot(edge_horizont, edge_vertical)
            #display in debbug window
            #datatoDisp_lab = segmentedM
            #datatoDisp_lab[segmentedEdges>0] = 255

            datatoDisp_lab[segmentedM>0] = (count_gtvs+1)*24
            datatoDisp_lab[seed] = 0
            #display edema limits on T1 & T2
            datatoDisp_T1[segmentedEdges>0,1] = 255

            datatoDisp_T2[segmentedM>0,randint(0,2)] = randint(180,230)
            #datatoDisp_T2[segmentedM>0,1] = int((randint(0,100)/100)*datatoDisp_T2[segmentedM>0,1])
            #datatoDisp_T2[segmentedM>0,2] = int((randint(0,100)/100)*datatoDisp_T2[segmentedM>0,2])
            
            #datatoDisp_T2[segmentedEdges>0,0] = randint(0,255)
            #datatoDisp_T2[segmentedEdges>0,1] = randint(0,255)
            #datatoDisp_T2[segmentedEdges>0,2] = randint(0,255)
            #seed display
            datatoDisp_T2[seed[0],seed[1],seed[2],0] = 255
            datatoDisp_T2[seed[0],seed[1],seed[2],1] = 0
            datatoDisp_T2[seed[0],seed[1],seed[2],2] = 0
            #count +1 gtv
            count_gtvs = count_gtvs+1

            #save edema and gtv's matrixs and volumes


    ###flag that the job is done and found one or more gtv(s)
    if count_gtvs > 0:
        flag_labeldone = 1

    print "Edema segemtation done from " + str(count_gtvs) + "GTV(s)"
    print numpy.sum(labelM)
    print numpy.sum(segmentedM)



################ autoregistration : if T1 and Flair are not well registered, try to realign FLAIR on T1
def autoregistration():
    global selT1
    global selT2
    global lstExams

    print "AUTO-REGISTRATION launched - this action overwrite affine matrixs"

##    def command_iteration(method) :
##        print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
##                                           method.GetMetricValue(),
##                                           method.GetOptimizerPosition()))
##
##    fixed = sitk.GetImageFromArray(lstExams[selT1]['data'])
##    fixed = sitk.Normalize(fixed)
##    fixed = sitk.DiscreteGaussian(fixed, 2.0)
##
##
##    moving = sitk.GetImageFromArray(lstExams[selT2]['data'])
##    moving = sitk.Normalize(moving)
##    moving = sitk.DiscreteGaussian(moving, 2.0)
##
##
##    R = sitk.ImageRegistrationMethod()
##
##    R.SetMetricAsJointHistogramMutualInformation()
##
##    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-5, convergenceWindowSize=5)
##
##    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
##
##    R.SetInterpolator(sitk.sitkLinear)
##
##    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
##
##    outTx = R.Execute(fixed, moving)
##
##    print("-------")
##    print(outTx)


############### SAVE WORK as CSV and save segmented mask matrixs in the patient's directory
def save():
    global patient
    global lstROIs
    global lstExams
    global selT1
    global dataRaw_T1
    global dataRaw_T2
    global t2seg_wind
    
    csv_dir = tkFileDialog.askdirectory()
    ############save data on patient
    #row to save (as list) : id, name, birthdate, gtvnames(separated by #), pixel width/height/..., gtvvols(#) and edemavs vol(#)
    mylist = [patient['id'], patient['name'], patient['birthdate'], lstExams[selT1]['PixelSpacing'][1], lstExams[selT1]['PixelSpacing'][2], lstExams[selT1]['PixelSpacing'][3]]
    gtv_names = ''
    gtv_vols = ''
    edema_vols = ''
    gtv_vtot = 0
    edema_vtot = 0
    sgt_level = t2seg_wind[1]#level of segmentaiton (windowing)
    sgt_width = t2seg_wind[0] #width of segemntation (windowing)
    Vvxl = lstExams[selT1]['PixelSpacing'][1]*lstExams[selT1]['PixelSpacing'][2]*lstExams[selT1]['PixelSpacing'][3]
    for ROIid in lstROIs:
         #foreach ROIs, if found 1 call "gtv..."
        if "gtv" in lstROIs[ROIid]['name'].lower(): # and count_gtvs == 0:
            if gtv_names != '':   #already a gtv known, so add a #
                gtv_names = gtv_names + '#'
                gtv_vols = gtv_vols + '#'
                edema_vols = edema_vols + '#'

            #add data to each string
            gtv_names = gtv_names + str(lstROIs[ROIid]['name'])
            gtv_vols = gtv_vols + str(lstROIs[ROIid]['GTVvol'])
            edema_vols = edema_vols + str(lstROIs[ROIid]['EDEMAvol'])

            #calculate total volume in mL for edema et gtv
            gtv_vtot = gtv_vtot + Vvxl*lstROIs[ROIid]['GTVvol']
            edema_vtot = edema_vtot + Vvxl*lstROIs[ROIid]['EDEMAvol']

            ##save a npy file for each segmentation
            numpy.save(os.path.join(csv_dir,'npy',str(patient['id'])+'_gtv_'+str(lstROIs[ROIid]['name']).lower()+'.npy'), lstROIs[ROIid]['GTVseg'])
            numpy.save(os.path.join(csv_dir,'npy',str(patient['id'])+'_edema_'+str(lstROIs[ROIid]['name']).lower()+'.npy'), lstROIs[ROIid]['EDEMAseg'])
                        
                
    mylist.append(gtv_names)
    mylist.append(gtv_vols)
    mylist.append(gtv_vtot)
    mylist.append(edema_vols)
    mylist.append(edema_vtot)
    mylist.append(sgt_level)
    mylist.append(sgt_width)

            
    with open(os.path.join(csv_dir,'MetaBrain.csv'), 'a') as myfile:
        
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(mylist)

    ################save voxel informations
    #export in cvs_dir/npy
    ## save raw dar from T1 and FLAIR
    numpy.save(os.path.join(csv_dir,'npy',str(patient['id'])+'_raw_T1.npy'), dataRaw_T1)
    numpy.save(os.path.join(csv_dir,'npy',str(patient['id'])+'_raw_FLAIR.npy'), dataRaw_T2)

    ##save current segmentation windowing ina CSV file
    script_path = os.path.dirname(os.path.abspath(__file__))
    mylist_sgt = [time.time(), sgt_level, sgt_width]
    with open(os.path.join(script_path,'sgt_windowing_history.csv'), 'a') as myfile_sgt:
        
        wr = csv.writer(myfile_sgt, quoting=csv.QUOTE_ALL)
        wr.writerow(mylist_sgt)

    print "Data saved"


############restart program by button
def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)
############################################
##########################################
##########################################
##########################################



##scroll mouse event
ScrollWcount = 0
    # with Windows OS
root.bind("<MouseWheel>", mouse_wheel)
    # with Linux OS
root.bind("<Button-4>", mouse_wheel)
root.bind("<Button-5>", mouse_wheel)

## click, drag and release B1 events (change T2 windowing)
root.bind("<B1-Motion>", t2windowing)
root.bind("<ButtonRelease-1>", mouse_release)
#######Buttons
# open patient's dir
btn_open=Button(root, text="Open patient ...", width=15
                , command=getPatientDir)
btn_open.place(x=1000, y=20)

#listbox T2 MPR
Label(root, text="Select T1 MPR...").place(x=1000, y=60)
lstbxT1=Listbox(root)
lstbxT1.place(x=1000, y=80)
lstbxT1.bind("<<ListboxSelect>>", displayT1)
#listbox T2 FLAIR
Label(root, text="Select T2 FLAIR...").place(x=1000, y=270)
lstbxT2=Listbox(root)
lstbxT2.place(x=1000, y=290)
lstbxT2.bind("<<ListboxSelect>>", displayT2)

# segmentation button
btn_segment=Button(root, text="Launch segmentation", width=15
                , command=segment_flair)
btn_segment.place(x=1000, y=500)

# auto_registration button
btn_segment=Button(root, text="Auto-registration", width=15
                , command=autoregistration)
btn_segment.place(x=1000, y=550)

# save button
btn_save=Button(root, text="Save", width=15
                , command=save)
btn_save.place(x=1000, y=600)

# restart button
btn_save=Button(root, text="restart", width=15
                , command=restart_program)
btn_save.place(x=1000, y=650)

######Display image
##photo = ImageTk.PhotoImage(file= r"/Users/yvanpin/Desktop/test.png")
##canvas = Tk.Canvas(root, width = 250, height = 250)
##canvas.pack(side=TOP, anchor=NW)
##canvas.create_image(0,0, anchor = Tk.NW, image=photo)

##Place elements


findMRI_ID('')

root.mainloop()

