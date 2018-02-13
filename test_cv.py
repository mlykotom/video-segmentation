import cv2

lol = cv2.imread("results/cityscapes_2_img.png")
print(lol.shape)

cv2.imshow("aho", lol)
cv2.waitKey()