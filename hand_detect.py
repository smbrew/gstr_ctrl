# -*- coding: utf-8 -*-
import cv2
import numpy as np

def main():
    fgbg = cv2.createBackgroundSubtractorKNN()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    b = 7
    #fgbg = cv2.createBackgroundSubtractorGMG()
    ret, bg = cap.read()
    gbg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    gbg_blur = cv2.GaussianBlur(gbg,(b,b),0)
    bg_blur = cv2.GaussianBlur(bg,(b,b),0)
    k=0
    while(True):
        if k==0:
            ret, bg = cap.read()
            gbg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            gbg_blur = cv2.GaussianBlur(gbg,(b,b),0)
            bg_blur = cv2.GaussianBlur(bg,(b,b),0)
        k = (k+1) % 60
        # Capture frame-by-frame
        ret, frame = cap.read()
        im = frame.copy()
        # Our operations on the frame come here
        gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gim_blur = cv2.GaussianBlur(gim,(b,b),0)
        im_blur = cv2.GaussianBlur(im,(b,b),0)
        
        res = cv2.absdiff(im_blur, bg_blur)
        gres = cv2.absdiff(gim_blur, gbg_blur)
        gres = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        ret, mres = cv2.threshold(gres, 25, 255, cv2.THRESH_BINARY)
        
        for i in range(0,3):
            ret, m = cv2.threshold(res[:,:,i], 25, 255, cv2.THRESH_BINARY)
            mres = cv2.max(mres, m)
        
        gres = cv2.cvtColor(gres, cv2.COLOR_GRAY2BGR)
        

        mres = cv2.cvtColor(mres, cv2.COLOR_GRAY2BGR)  
        mres = cv2.morphologyEx(mres, cv2.MORPH_OPEN, kernel)
        mres = cv2.morphologyEx(mres, cv2.MORPH_CLOSE, kernel)
        
        
        fgmask = fgbg.apply(im)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        # Display the resulting frame
        
        
        fg = im.astype(float)
        alpha = fgmask.astype(float)/100        
        fg = cv2.multiply(alpha, fg)/255
        
        stacked = np.hstack((fg, mres))
        cv2.imshow('frame', stacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    