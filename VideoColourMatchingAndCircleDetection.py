from multiprocessing.sharedctypes import Value
import cv2
import numpy as np


def nothing(x): # apparently needs to be here to use trackbar 
    pass

if __name__ == '__main__':
    cap = cv2.VideoCapture(0) # choose camera

    # Trackbars er grafiske sliders der sætter værdier.
    cv2.namedWindow('HSV and HughCircle Values') # Create window with trackbars
    cv2.createTrackbar("H MIN", "HSV and HughCircle Values", 60, 179, nothing) # Trackbar object, params: (Label/Name, Window name (Which window it belongs to), default value, max. value, callback option)
    cv2.createTrackbar("S MIN", "HSV and HughCircle Values", 120, 255, nothing)
    cv2.createTrackbar("V MIN", "HSV and HughCircle Values", 60, 255, nothing)
    cv2.createTrackbar("H MAX", "HSV and HughCircle Values", 179, 179, nothing)
    cv2.createTrackbar("S MAX", "HSV and HughCircle Values", 255, 255, nothing)
    cv2.createTrackbar("V MAX", "HSV and HughCircle Values", 255, 255, nothing)
    cv2.createTrackbar("Hough param1", "HSV and HughCircle Values", 192, 300, nothing)
    cv2.createTrackbar("Hough param2", "HSV and HughCircle Values", 10, 15, nothing)

    while True:

        ret, frame = cap.read() # get frame from webcam

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)# Copy of frame, in HSV colourspace
        imgCopy = frame.copy() # copy of frame used later for showing circle detection in imshow()

        # using trackbar values in variables
        h_min = cv2.getTrackbarPos("H MIN", "HSV and HughCircle Values") 
        s_min = cv2.getTrackbarPos("S MIN", "HSV and HughCircle Values")
        v_min = cv2.getTrackbarPos("V MIN", "HSV and HughCircle Values")
        h_max = cv2.getTrackbarPos("H MAX", "HSV and HughCircle Values")
        s_max = cv2.getTrackbarPos("S MAX", "HSV and HughCircle Values")
        v_max = cv2.getTrackbarPos("V MAX", "HSV and HughCircle Values")

        #settin lower and upper values to be used for mask. (3 channels, so 3 values needed)
        lowerVal = np.array([h_min, s_min, v_min]) 
        upperVal = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lowerVal, upperVal) # mask based on min. and max. HSV values
        result = cv2.bitwise_and(frame, frame, mask=mask)

        hsv_min = "MIN H:{} S:{} V:{}".format(h_min,s_min,v_min) # Formatting for text showing min. and max. HSV values in frame
        hsv_max = "MAX H:{} S:{} V:{}".format(h_max, s_max, v_max)

        cv2.putText(frame, hsv_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # add text to frame
        cv2.putText(frame, hsv_max, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        

        ################################## Hugh circle detection ##################################

        hughParam1 = float(cv2.getTrackbarPos("Hough param1", "HSV and HughCircle Values"))/100 # read params from trackbar
        hughParam2 = cv2.getTrackbarPos("Hough param2", "HSV and HughCircle Values")        

        gray = cv2.medianBlur(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),5) 
        circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, hughParam1, hughParam2)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(imgCopy, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(imgCopy, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        ############################################################################################

        #cv2.imshow('HSV', hsv) # Show the image in HSV colourspace
        cv2.imshow("HSV and HughCircle Values", result) # show trackbars and frame
        #cv2.imshow('frame', result)
        cv2.imshow('mask', mask) # Shows the mask
        cv2.imshow('HughCirle detection', imgCopy) # show copy of frame + hughcircles
    
        if cv2.waitKey(20) == 27: # wait for keyinput. (27 = ESC key)
            break
cap.release() # release camera (camera cannot be used for other applications while running this program)
cv2.destroyAllWindows # Close all the windows