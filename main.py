import cv2
import matplotlib.pyplot as plt
import face_location
import removeBG_API_request
import smile_detection
import face_relighting
import numpy as np


image = cv2.imread('./photos/WhatsApp Image 2020-11-21 at 14.21.25.jpeg')
h, w = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_loc = face_location.get_face_location(gray)
face_percentage = ((face_loc[2] - face_loc[0]) * (face_loc[3] - face_loc[1])) / (h * w) * 100
smile_coeff = smile_detection.detect_smile(gray, face_loc)
smile = 'none'
if smile_coeff == 2:
    smile = 'too much'
elif smile_coeff == 1:
    smile = 'fine'

canvas = image.copy()
cv2.putText(canvas, f'Face occupying the {round(face_percentage)}% (recom. 20%).', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 60, 0), 3)
cv2.putText(canvas, f'Smile: {smile}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 60, 0), 3)

cv2.imwrite('./results/result_phase_1.png', canvas)

relighted = face_relighting.relight(image, is_bgr=True)
cv2.imwrite('./results/result_phase_2.png', relighted)

mask = removeBG_API_request.remove_background(relighted, image_format='.jpg', is_RGB=False)[:, :, 3]
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.imwrite('./results/result_phase_3.png', mask)

h_div_w = 1.231
new_w = w
new_h = int(w * h_div_w)
output = cv2.getRectSubPix(relighted, (new_w, new_h), (w / 2, h / 2))
mask = cv2.getRectSubPix(mask, (new_w, new_h), (w / 2, h / 2))

white = np.ones_like(output) * 255

result = (white * (1 - (mask / 255.)) + (mask / 255.) * output).astype('uint8')

cv2.imwrite('./results/result_last_phase.png', result)

fig, axs = plt.subplots(3, 3)
fig.suptitle('Diego Bonilla auto-photo-ID')

axs[0, 0].imshow(white)
axs[0, 0].axis('off')
axs[2, 0].imshow(white)
axs[2, 0].axis('off')

axs[1, 0].imshow(image[:, :, ::-1])
axs[1, 0].set_title('Original Photo')
axs[1, 0].axis('off')

axs[0, 1].imshow(canvas[:, :, ::-1])
axs[0, 1].set_title('Photo Info')
axs[0, 1].axis('off')

axs[1, 1].imshow(relighted[:, :, ::-1])
axs[1, 1].set_title('Relighted Photo')
axs[1, 1].axis('off')

axs[2, 1].imshow(mask)
axs[2, 1].set_title('Person Mask')
axs[2, 1].axis('off')

axs[1, 2].imshow(result[:, :, ::-1])
axs[1, 2].set_title('Result')
axs[1, 2].axis('off')

axs[0, 2].imshow(white)
axs[0, 2].axis('off')
axs[2, 2].imshow(white)
axs[2, 2].axis('off')

plt.show()
