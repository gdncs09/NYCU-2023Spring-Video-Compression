import cv2
import matplotlib.pyplot as plt
import numpy as np

#Input
originalImg = cv2.imread('lena.png')
(B, G, R) = cv2.split(originalImg)
originalImg = cv2.merge((R,G,B))

#Calculate YUV
Y = np.uint8(0.299*R + 0.587*G+ 0.114*B)
U = np.uint8(-0.169*R - 0.331*G + 0.5*B + 128)
V  = np.uint8(0.5*R - 0.419*G - 0.081*B + 128)

#Calculate CbCr
Cb = np.uint8(-0.168736*R - 0.331264*G + 0.5*B + 128)
Cr = np.uint8(0.5*R - 0.418688*G - 0.081312*B + 128)

#Output
#cv2.imwrite('R.jpg', R)
#cv2.imwrite('G.jpg', G)
#cv2.imwrite('B.jpg', B)
#cv2.imwrite('Y.jpg', Y)
#cv2.imwrite('U.jpg', U)
#cv2.imwrite('V.jpg', V)
#cv2.imwrite('Cb.jpg', Cb)
#cv2.imwrite('Cr.jpg', Cr)

fig, ax = plt.subplots(3, 3)
ax[0][0].axis('off')
ax[0][0].set_title('R')
ax[0][0].imshow(R, cmap = 'gray')

ax[0][1].axis('off')
ax[0][1].set_title('G')
ax[0][1].imshow(G, cmap = 'gray')

ax[0][2].axis('off')
ax[0][2].set_title('B')
ax[0][2].imshow(B, cmap = 'gray')

ax[1][0].axis('off')
ax[1][0].set_title('Y')
ax[1][0].imshow(Y, cmap = 'gray')

ax[1][1].axis('off')
ax[1][1].set_title('U')
ax[1][1].imshow(U, cmap = 'gray')

ax[1][2].axis('off')
ax[1][2].set_title('V')
ax[1][2].imshow(V, cmap = 'gray')

ax[2][0].axis('off')
ax[2][0].set_title('Original')
ax[2][0].imshow(originalImg)

ax[2][1].axis('off')
ax[2][1].set_title('Cb')
ax[2][1].imshow(Cb, cmap = 'gray')

ax[2][2].axis('off')
ax[2][2].set_title('Cr')
ax[2][2].imshow(Cr, cmap = 'gray')
plt.show()

    
