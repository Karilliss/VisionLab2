import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_gabor_filters():
    """Створює банк фільтрів Габора з різними орієнтаціями та масштабами."""
    filters = []
    ksize = 31  

    for theta in np.arange(0, np.pi, np.pi / 4):  
        for sigma in [3.0, 5.0]:                 
            for lambd_val in [10.0, 20.0]:      
                kern = cv2.getGaborKernel(
                    ksize=(ksize, ksize),
                    sigma=sigma,
                    theta=theta,
                    lambd=lambd_val, 
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                kern /= 1.5 * kern.sum()  
                filters.append(kern)
                
    return filters

def apply_gabor_bank(img, filters):
    """
    Застосовує банк фільтрів Габора до зображення, 
    вираховує статистичні ознаки (середнє та сер. відхилення) та повертає їх вектор.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    responses = []
    
    for kern in filters:
        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern)
        responses.append(fimg)

        features.append(np.mean(fimg))
        features.append(np.std(fimg))
        
    return np.array(features), responses

def compare_with_gabor_bank(img_path1, img_path2):
    print("=" * 65)
    print("Порівняння знімків за банком фільтрів Габора (текстурний аналіз)")
    print("=" * 65)
    
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Не вдалося завантажити одне з зображень. Перевірте назви файлів!")
        
    filters = build_gabor_filters()
    print(f"Сформовано банк фільтрів Габора: {len(filters)} фільтрів.")
    
    feat1, resp1 = apply_gabor_bank(img1, filters)
    feat2, resp2 = apply_gabor_bank(img2, filters)

    feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-6)
    feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-6)
    
    cosine_similarity = np.dot(feat1_norm, feat2_norm)
    euclidean_dist = np.linalg.norm(feat1 - feat2)
    
    print(f"Знімок 1: {img_path1}")
    print(f"Знімок 2: {img_path2}")
    print("-" * 65)
    print(f"Косинусна схожість вектору ознак = {cosine_similarity:.4f}  (ідеал: 1.0)")
    print(f"Евклідова відстань між ознаками  = {euclidean_dist:.4f}  (чим менше, тим краще)")
    print("=" * 65)
    
    if cosine_similarity > 0.75:
        print("\n[РЕЗУЛЬТАТ]: Текстурні структури та орієнтації об'єктів високого рівня збігаються.")
        print("Придатність ознак успішно доведено за допомогою банку фільтрів Габора!")
    print("=" * 65)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Знімок 1: {img_path1}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Знімок 2: {img_path2}")
    axes[0, 1].axis('off')
 
    axes[1, 0].imshow(resp1[0], cmap='gray')
    axes[1, 0].set_title("Відгук фільтра Габора (Знімок 1)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(resp2[0], cmap='gray')
    axes[1, 1].set_title("Відгук фільтра Габора (Знімок 2)")
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Порівняння текстур (Косинусна схожість: {cosine_similarity:.3f})", fontsize=13)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_with_gabor_bank('titicaca_bing.png', 'titicaca_mapcarta1.png')
