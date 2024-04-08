import sys
import os
import numpy as np
import cv2 as cv

def cell_count(drawing):
    
    gray = cv.cvtColor(drawing, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    
    # Find contours and draw them on the original image
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    counter = 0

    # In this loop, we calculate the rectangle area containing the cell
    # Area must be >100 and <1250
    for c in cnts:        
        x,y,w,h = cv.boundingRect(c)
        if ((w*h)>100) & ((w*h)<1250):
            counter += 1
            # Drawing 
            cv.drawContours(drawing, [c], -1, (255, 255, 255), -1)
    
    for y in range(0, len(drawing)):
        for x in range(0, len(drawing[0])):
            b = drawing.item(y, x, 0)
            
            if b == 0:
                drawing.itemset((y, x, 0), 0)
                drawing.itemset((y, x, 1), 0)
                drawing.itemset((y, x, 2), 0) 

    print (f"There are {counter} cells on the image.")

    return counter

def classifyTissue(image):

    B = image[:,:,0]
    R = image[:,:,2]
    V = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:,:,2]
    

    kernel7 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    kernel9 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
    kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    kernel9 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))

    
    background = (V < 220).astype(np.uint8)
    background = cv.erode(background, kernel7)
    background = cv.dilate(background, kernel7)
    background = cv.dilate(background, kernel9)
    mask_background=background*255

    # cnts, _ = cv.findContours(mask_background, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print("contours : ", len(cnts))
    # # Establece el área mínima para filtrar los contornos
    # area_minima = 100_000_000 # Ajusta este valor según tus necesidades
    # # Itera sobre los contornos y filtra aquellos con un área menor que el umbral
    # for contour in cnts:
    #     area = cv.contourArea(contour)
    #     if area < area_minima:
    #         # Si el área es menor que el umbral, píntalo de blanco
    #         cv.drawContours(mask_background, [contour], -1, 255, thickness=cv.FILLED)

    # Areas con "mucho" color o con mas azul que rojo. Creación de una máscara donde los valores en V son mayores a 150 o donde B sea mayor que R
    stroma = ((V > 150) | (B > R)).astype(np.uint8)
    stroma = cv.erode(stroma,  kernel5)
    stroma = cv.dilate(stroma, kernel9)
    stroma = cv.erode(stroma, kernel5)
    stroma_mask = np.where(stroma == 0, 50, 75 * stroma) 

    # Find the nuclei based on the relation between blue and red, value level and the subtraction of the stroma. Creación de una máscara donde los valores en V son mayores a 100
    mask1 = (V > 100).astype(np.uint8)
    # Creación de una máscara donde los valores en 1.75 * B sea mayor que R
    mask2 = (1.8 * B > R).astype(np.uint8)
    not_stroma = ~stroma
    threshold = 254
    not_stroma = (not_stroma > threshold).astype(np.uint8)
    not_stroma=not_stroma*255

    combined_mask = (mask1 & not_stroma & mask2)
    nuclei = cv.erode(combined_mask, kernel5)
    nuclei = cv.dilate(nuclei, kernel5)
    nuclei_mask = 255 * nuclei


    mask = np.where((mask_background== 0), mask_background, stroma_mask)
    mask = np.where((mask == 75) | (mask == 0), mask, 150)
    mask = np.maximum(mask,nuclei_mask)
            
    hist = cv.calcHist([mask],[0],None,[256],[0,256])
    print("\033[1;32mTotal area:\033[0m")
    print(f"    Tissue: {0.21e-6*hist[150][0]} mm^2")
    print(f"    Nuclei: {0.21e-6*hist[255][0]} mm^2")
    print(f"    Stroma: {0.21e-6*hist[75][0]} mm^2\n") #falla, salen nuclei y stroma iguales
    return mask

def main():
    try:
        print("\n\033[32m╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮"
              "\n│              🧬  CELL DETECTOR  🧬             │"
              "\n╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯\033[0m\n")
        image_file = input("Introduce a tissue image: ")
        if not os.path.isfile(image_file):
            print(f"\n\033[31m'{image_file}' does not exist.\033[0m\n")
            sys.exit(1)
            
        img = cv.imread(image_file)
        if img is None:
            print(f"\n\033[31m'{image_file}' doesn't contain an image.\033[0m\n")
            sys.exit(1)


        # Image size
        print(f"\nAnalizing '{image_file}'\n")
        print(f"\tImage size: {img.shape[0]} x {img.shape[1]} pixels, ({img.shape[2]} channels).")
        print(f"\tImage size: {img.shape[0] / 2100} x {img.shape[1] / 2100} mm") 

        max_iter = 10
        epsilon = 1.0
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
        # Avoid problems with points
        flags = cv.KMEANS_RANDOM_CENTERS

        # Data as a matrix
        matrix = np.float32(img.flatten().reshape(img.shape[0]*img.shape[1],3))

        # Best K                                                    
        for K in range(2,3):
            print(f"K: {K}")
            compactness, labels, centers = cv.kmeans(matrix, K, None, criteria, max_iter, flags)
            centers = np.uint8(centers)
            clustered_image = centers[labels.flatten()]
            clustered_image = clustered_image.reshape(img.shape)  

        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]


        # Using only Blue and Green channels for the nuclei
        srcBG = B+G
        # Not overflowing the uitn8 type
        srcBG[srcBG < np.minimum(B,G)] = 255
        th, dst = cv.threshold(srcBG, 0, 255, cv.THRESH_OTSU)

        cell_mask = np.zeros(img.shape, np.uint8)
        contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(cell_mask, contours, -1, (0,0,255), 2)

        mask=classifyTissue(img)

        # For not overriding 'mask'
        mask_show = mask

        if max(mask.shape[:2]) > 1000:
            mask_show = cv.resize(mask, (0, 0), fx=500/max(mask.shape[:2]), fy=500/max(mask.shape[:2]))
        cv.imshow('Tissue, nuclei and stromas.', mask_show)
        cv.waitKey(0)

        counter = cell_count(cell_mask)
        if max(mask.shape[:2]) > 1000:
            cell_mask = cv.resize(cell_mask, (0, 0), fx=600/max(cell_mask.shape[:2]), fy=600/max(cell_mask.shape[:2]))
        cv.imshow('Cells.', cell_mask)
        cv.waitKey(0)

    except Exception as e:
        print("Error on main:", e)

if __name__ == '__main__':
    main()