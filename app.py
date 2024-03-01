import streamlit as st
import cv2
import numpy as np
import utlis
import copy

#while(True)
st.title('OMG Grading')

cam= st.selectbox('Select Camera',['Select','Yes','No'],index=0)
#cam = st.text_input("Do you want to use camera? (y/n): ") # if y, then True else False
if cam.lower() == "yes":
    cam=True
    path=None
else:
    cam=False
    path = st.text_input("Enter the path of video file: ")# If cam is false, enter the image path here

NO_q = st.number_input("Enter number of Questions to be graded. ", min_value=0, max_value=10, step=1)
NO_c = st.number_input("Enter number of Choices each question has? ", min_value=0, max_value=10, step=1)
ans_key = []
for i in range(1, NO_q+1):
    ans = st.number_input("Enter answer key for Q"+str(i)+": ", min_value=0, max_value=10, step=1)  # Example : For Q5 answer key will be 5
    ans_key.append(ans-1)

########################################################################
webCamFeed = cam
pathImage = path
heightImg = 700
widthImg  = 700
questions= NO_q
choices= NO_c
ans= ans_key
########################################################################

count=0
frame_placeholder = st.image([])
stop_button_pressed = st.button("Stop")

if st.button('Start Grading'):
    if cam:
        
        cap = cv2.VideoCapture(0)
        cap.set(10,160)
    
    while (webCamFeed or path) and not stop_button_pressed:
    
        if webCamFeed: _, img = cap.read()
        else:img = cv2.imread(pathImage)
        try:
            img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
        except:
            pass
        imgFinal = copy.copy(img)
        imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
        imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 

        try:
            ## FIND ALL COUNTOURS
            imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
            imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
            contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
            rectCon = utlis.rectContour(contours) # FILTER FOR RECTANGLE CONTOURS
            biggestPoints= utlis.getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
            gradePoints = utlis.getCornerPoints(rectCon[1]) # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

            

            if biggestPoints.size != 0 and gradePoints.size != 0:

                # BIGGEST RECTANGLE WARPING
                biggestPoints=utlis.reorder(biggestPoints) # REORDER FOR WARPING
                cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
                pts1 = np.float32(biggestPoints) # PREPARE POINTS FOR WARP
                pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
                matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE

                # SECOND BIGGEST RECTANGLE WARPING
                cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20) # DRAW THE BIGGEST CONTOUR
                gradePoints = utlis.reorder(gradePoints) # REORDER FOR WARPING
                ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
                ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
                matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)# GET TRANSFORMATION MATRIX
                imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # APPLY WARP PERSPECTIVE

                # APPLY THRESHOLD
                imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
                imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE

                boxes = utlis.splitBoxes(imgThresh) # GET INDIVIDUAL BOXES
                #cv2.imshow("Split Test ", boxes[3])
                countR=0
                countC=0
                myPixelVal = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
                for image in boxes:
                    #cv2.imshow(str(countR)+str(countC),image)
                    totalPixels = cv2.countNonZero(image)
                    myPixelVal[countR][countC]= totalPixels
                    countC += 1
                    if (countC==choices):countC=0;countR +=1

                # FIND THE USER ANSWERS AND PUT THEM IN A LIST
                myIndex=[]
                for x in range (0,questions):
                    arr = myPixelVal[x]
                    myIndexVal = np.where(arr == np.amax(arr))
                    myIndex.append(myIndexVal[0][0])
                #print("USER ANSWERS",myIndex)

                # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
                grading=[]
                for x in range(0,questions):
                    if ans[x] == myIndex[x]:
                        grading.append(1)
                    else:grading.append(0)
                #print("GRADING",grading)
                score = (sum(grading)/questions)*100 # FINAL GRADE
                #print("SCORE",score)

                # DISPLAYING ANSWERS
                utlis.showAnswers(imgWarpColored,myIndex,grading,ans) # DRAW DETECTED ANSWERS
                utlis.drawGrid(imgWarpColored) # DRAW GRID
                imgRawDrawings = np.zeros_like(imgWarpColored) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
                utlis.showAnswers(imgRawDrawings, myIndex, grading, ans) # DRAW ON NEW IMAGE
                invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
                imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP

                # DISPLAY GRADE
                imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # NEW BLANK IMAGE WITH GRADE AREA SIZE
                cv2.putText(imgRawGrade,str(int(score))+"%",(70,100)
                            ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3) # ADD THE GRADE TO NEW IMAGE
                invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # INVERSE TRANSFORMATION MATRIX
                imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # INV IMAGE WARP

                # SHOW ANSWERS AND GRADE ON FINAL IMAGE
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

        except:
            pass
        
        frame_placeholder.image(imgFinal, channels='BGR')

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

# RELEASE RESOURCES BE

    try:
        cv2.destroyAllWindows()
        cap.release()
    except:
        pass
    #