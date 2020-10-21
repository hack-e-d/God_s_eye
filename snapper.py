import cv2 

def snap(ptc,ptf,img):
    dim=(2000,1290)
    img = cv2.rectangle(img, (ptc[1]-1, ptc[0]-1), (ptc[1]+1,ptc[0]+1), (0, 255, 0), 2)
    img = cv2.rectangle(img, (ptf[1]-1, ptf[0]-1), (ptf[1]+1,ptf[0]+1), (0, 0, 255), 2)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    cv2.imwrite("colision.jpg",resized)