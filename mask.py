import cv2
import numpy as np


def create_mask(image):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    return mask

image = cv2.imread(r'C:\Users\Computer_PA24\Downloads\Projects\AUV Project\Arrow_Detection\up.jpg')



blurred_image = cv2.GaussianBlur(image,(5,5),0)


arrow_mask = create_mask(blurred_image)

cv2.imwrite('arrow_mask.jpg',arrow_mask)



contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



contour_image = np.zeros_like(blurred_image)

blurred_image_copy=blurred_image.copy()



cv2.drawContours(blurred_image_copy, contours, -1, (0,255,0), 2)
cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)
height, width, _ = blurred_image.shape


for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    cv2.rectangle(blurred_image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.rectangle(contour_image,(x, y), (x + w, y + h), (255, 0, 0), 2)





cv2.imshow('Blurred_image',blurred_image)
cv2.imshow('Mask', arrow_mask)
cv2.imshow('Detection_image', blurred_image_copy)
cv2.imshow('Detection',contour_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
