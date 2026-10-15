import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_fully_automated_research(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не вдалося завантажити файл: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    negative = cv2.bitwise_not(gray) 

    kernel_sepia = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel_sepia)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    sepia_rgb = cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    canny_edges = cv2.Canny(blurred, 40, 120)

    gabor_responses = []
    for theta in np.arange(0, np.pi, np.pi / 4): 
        kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(blurred, cv2.CV_32F, kernel)
        gabor_responses.append(filtered)
    
    gabor_filtered = np.max(np.array(gabor_responses), axis=0)
    gabor_filtered = cv2.normalize(gabor_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, gabor_filtered = cv2.threshold(gabor_filtered, 40, 255, cv2.THRESH_BINARY)
    gabor_filtered = cv2.bitwise_not(gabor_filtered) 

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_water = np.array([0, 0, 0])
    upper_water = np.array([180, 255, 75])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_open)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel_close)

    border = 50
    water_mask[:border, :] = 0
    water_mask[-border:, :] = 0
    water_mask[:, :border] = 0
    water_mask[:, -border:] = 0
    
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1000 < area < 0.6 * (h_img * w_img):
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if 0.2 * w_img < cX < 0.7 * w_img and 0.1 * h_img < cY < 0.9 * h_img:
                        valid_contours.append(c)
                        
        if valid_contours:
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
            for c in valid_contours[:2]:
                cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 3)

    plt.figure(figsize=(15, 9))
    
    plots = [
        (img_rgb, "Оригінал (RGB)", None),
        (gray, "Відтінки сірого", 'gray'),
        (negative, "Негатив", 'gray'),
        (sepia_rgb, "Сепія", None),
        (enhanced, "CLAHE Корекція", 'gray'),
        (canny_edges, "Векторизація: Canny", 'gray'),
        (gabor_filtered, "Банк фільтрів Габора", 'gray'),
        (contour_img, "Автоматичний контур (Contours)", None)
    ]
    
    for i, (img_data, title, cmap_type) in enumerate(plots):
        plt.subplot(2, 4, i + 1)
        if cmap_type:
            plt.imshow(img_data, cmap=cmap_type)
        else:
            plt.imshow(img_data)
        plt.title(title, fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    perform_fully_automated_research('titicaca_bing.png')


