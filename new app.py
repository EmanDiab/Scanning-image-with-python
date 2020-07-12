from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils



# =============== Functions to help ============================
# function to order points to proper rectangle
def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# function to transform image to four points
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def findLargestCountours(cntList, cntWidths):
    newCntList = []
    newCntWidths = []

    # finding 1st largest rectangle
    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[first_largest_cnt_pos])
    newCntWidths.append(cntWidths[first_largest_cnt_pos])

    # removing it from old
    cntList.pop(first_largest_cnt_pos)
    cntWidths.pop(first_largest_cnt_pos)

    # finding second largest rectangle
    seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[seccond_largest_cnt_pos])
    newCntWidths.append(cntWidths[seccond_largest_cnt_pos])

    # removing it from old
    cntList.pop(seccond_largest_cnt_pos)
    cntWidths.pop(seccond_largest_cnt_pos)

    return newCntList, newCntWidths

image = cv2.imread('e.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)  # 11  //TODO 11 FRO OFFLINE MAY NEED TO TUNE TO 5 FOR ONLINE
gray = cv2.medianBlur(gray, 5)
edged = cv2.Canny(gray, 30, 400)

countours, hierarcy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
imageCopy = image.copy()
# approximate the contour
cnts = sorted(countours, key=cv2.contourArea, reverse=True)
screenCntList = []
scrWidths = []
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)  # cnts[1] always rectangle O.o
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    screenCnt = approx

    if len(screenCnt) == 4:
        (X, Y, W, H) = cv2.boundingRect(cnt)
        screenCntList.append(screenCnt)
        scrWidths.append(W)



screenCntList, scrWidths = findLargestCountours(screenCntList, scrWidths)


if not len(screenCntList) >= 2:  # there is no rectangle found
    print("No rectangle found")
elif scrWidths[0] != scrWidths[1]:  # mismatch in rect
    print("Mismatch in rectangle")


pts = screenCntList[0].reshape(4, 2)
rect = order_points(pts)


warped = four_point_transform(orig, screenCntList[0].reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

    # show the original and warped images

cv2.imshow("Original", image)
#imutils.resize(warped, height = 650)
cv2.imshow("warp", warped )
cv2.waitKey(0)

