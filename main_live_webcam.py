#importing custom built packages
import cv2 as cv
import tkinter as tk
import distance_calculator as dc
import detector as d
import distraction_system as ds
import snapper as sp
import sys
from tkinter import messagebox
import time 

print("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\tGODs EYE >>>WELCOME\n\n\n\n\n\n")
vardis=0

def close():
    print("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\tGODs EYE >>>THANKS FOR USING\n\n\n\n\n\n")
    sys.exit()
    
def hi():
    #function to process each frame 
    def done(img):
        copy_img=img
        fpoints=[0,0]
        cpoints=[0,0]
        vardis=0
        #finting location of child
        cpoints,fpoints=d.detect(img)
        if((cpoints[0]!=0 or cpoints[1]!=0) and (fpoints[0]!=0 or fpoints[1]!=0)):
            vardis = dc.distance(fpoints[0],fpoints[1],cpoints[0],cpoints[1])
            print("Distance of fork from child is ",vardis) 
            if(vardis<100 and vardis!=0):
                warning_message="the child is "+str(int(vardis))+" cm close to the fork"
                sp.snap(cpoints,fpoints,copy_img)
                ds.distract()
                messagebox.showinfo("warning",warning_message)
        
        
    cap=cv.VideoCapture(0)

    print("GODs EYE >>>Starting System")
    print("GODs EYE >>>Fetching Video")
    while(True):
        ret,frame=cap.read()
        if(ret == False):
            break
        done(frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    print("GODs EYE >>>Terminating Video")
top = tk.Tk() 
var='Press Q to exit'
top.wm_title("GOD'S EYE")
top.geometry("300x200") 
tk.Label( top, text=var ).place(x=92,y=150)
tk.Button(top, text = "CLOSE",activebackground = "pink", activeforeground = "blue",command=close).place(x = 100, y = 110)
tk.Button(top, text = "START",activebackground = "pink", activeforeground = "blue",command=hi).place(x = 100, y = 70)
top.mainloop() 