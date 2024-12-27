#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from keras.api.models import load_model
import imutils 
from matplotlib import pyplot as plt   
import os    
from PIL import Image   

def algo():
    input_size=48 
    def findBoard(img): #finding largest area using countour lines  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 20, 20) #removes noise
        edged = cv2.Canny(bfilter, 30, 180) #detecting edges
        kp = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #finding continuous points in the image
        contours = imutils.grab_contours(kp)
        newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
        plt.imshow(newimg)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15] #taking largest 15 contours to increase accuracy
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 15, True) #Checks if its a rectangular region
            if len(approx) == 4: #if length = 4 then rect
                location = approx
                break
        result = get_perspective(img, location)
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return result, location

    def get_perspective(img,location):
        h=900
        w=900
        #change the orientation and crop to board:
        points1=np.float32([location[0], location[3], location[1], location[2]])
        points2= np.float32([[0,0], [w,0],[0,h],[w,h]])


        #store transformation in matrix:
        matrix = cv2.getPerspectiveTransform(points1,points2)
        result = cv2.warpPerspective(img,matrix,(w,h))
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        
        return result # birds eye view of the board from an angled perspective




    #SPLITTING BOARD INTO 81 INDIVIDUAL IMAGES (9x9 MATRIX)
    def splitBoxes(board):
        rows = np.vsplit(board,9) #VERTICAL SPLIT 
        boxes = []
        for r in rows:
            cols = np.hsplit(r,9)# HORIZONTAL SPLIT
            for box in cols:
                box = cv2.resize(box, (input_size, input_size))/255.0            
                boxes.append(box)
        return boxes
    full_path = os.path.join('static/uploadedSudoku', os.listdir('static/uploadedSudoku')[0])
    img = cv2.imread(full_path)

    board, location = findBoard(img)
    plt.imshow(board)
    
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = splitBoxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)
    classes = np.arange(0, 10)
    model = load_model('model-OCR.h5')
    prediction = model.predict(rois)
    predicted_numbers = []
    for i in prediction:
        index = (np.argmax(i))
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)
    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)
    print(board_num)

    #Sumedh's solving algorithm ---------------------------------------------------
    import random

    def legal_state(board):
        for row in board:                                  # Checking validity of each row
            seen_row = []
            for element in row:
                if element in seen_row:
                    return False
                else:
                    if element != 0:
                        seen_row.append(element)

        for i in range(9):                                 # Checking validity of each column
            seen_column = []
            for k in range(9):
                if board[k][i] in seen_column:
                    return False
                else:
                    if board[k][i] != 0:
                        seen_column.append(board[k][i])

        for box_row_start in range(0, 9, 3):               # Checking validity of each 3x3 box
            for box_column_start in range(0, 9, 3):
                seen_box = []
                for i in range(box_row_start, box_row_start + 3):
                    for j in range(box_column_start, box_column_start + 3):
                        if board[i][j] != 0:
                            if board[i][j] in seen_box:
                                return False
                            else:
                                seen_box.append(board[i][j])
        return True



    def solve_sudoku(board):                       # Generate a random Sudoku board with empty spaces
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    random_numbers = list(range(1, 10))
                    random.shuffle(random_numbers)
                    for num in random_numbers:
                        board[row][col] = num
                        if legal_state(board) and solve_sudoku(board):
                            return True
                        board[row][col] = 0
                    return False
        return True
    solve_sudoku(board_num)
    print(board_num)
    #------------------------------------------------------------------------------
    #PRINTING THE SOLVED BOARD ON TOP OF ORIGINAL
    def showSolution(img, numbers,  color=(255,0,0)):
        W = int(img.shape[1]/9)
        H = int(img.shape[0]/9)
        for i in range (9):
                for j in range (9):
                    if numbers[(j*9)+i] !=0:
                        cv2.putText(img, str(numbers[(j*9)+i]),
                            (i*W+int(W/2)-int((W/4)),int((j+0.7)*H)),cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
        return img
    def invertperspective(img, mask, loc, h=900,w=900): #takes original image and creates a new plane to project on
        #same algo as getperspective function
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([location[0], location[3], location[1], location[2]])
        #once again applying perspective transform and warp
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(mask, matrix, (img.shape[1],img.shape[0]))
        return result

    binArr = np.where(np.array(predicted_numbers)>0, 0, 1) #binary array of 0 to 1 if >0 then added as 1
    print(binArr)
    flat_solved_board_nums = board_num.flatten()*binArr #get only solved numbers
    # create a mask
    mask = np.zeros_like(board) #mask of same size as board 
    solved_board_mask = showSolution(mask, flat_solved_board_nums) #prints solved numbers on new plane
    plt.imshow(solved_board_mask)


    inv = invertperspective(img, solved_board_mask, location)
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
    plt.imshow(combined)
    img = Image.fromarray(combined)
    os.remove(os.path.join('static/uploadedSudoku', os.listdir('static/uploadedSudoku')[0]))
    img.save('static/solvedSudoku/solved.jpg')





