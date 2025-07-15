import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def extract_color_features_from_segmented_region(img_bgr, mask):
    if img_bgr is None or mask is None:
        return None

    if mask.dtype != np.bool_:
        mask = mask > 0

    if img_bgr.shape[:2] != mask.shape[:2]:
        return None

    masked_bgr_pixels = img_bgr[mask]

    if masked_bgr_pixels.size == 0:
        return None

    temp_img_rgb = np.zeros_like(img_bgr)
    temp_img_hsv = np.zeros_like(img_bgr)

    temp_img_rgb[mask] = cv2.cvtColor(masked_bgr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    temp_img_hsv[mask] = cv2.cvtColor(masked_bgr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    avg_r = np.mean(temp_img_rgb[mask][:, 0])
    avg_g = np.mean(temp_img_rgb[mask][:, 1])
    avg_b = np.mean(temp_img_rgb[mask][:, 2])

    avg_h = np.mean(temp_img_hsv[mask][:, 0])
    avg_s = np.mean(temp_img_hsv[mask][:, 1])
    avg_v = np.mean(temp_img_hsv[mask][:, 2])

    return np.array([avg_r, avg_g, avg_b, avg_h, avg_s, avg_v])

def create_dummy_dataset(num_samples_per_class=100):
    data = []
    labels = []
    class_names = ['matang', 'mengkal', 'mentah'] 
    
    for _ in range(num_samples_per_class):
        data.append([np.random.uniform(220, 255), np.random.uniform(140, 200), np.random.uniform(0, 50), 
                     np.random.uniform(0, 20), np.random.uniform(180, 255), np.random.uniform(200, 255)])
        labels.append(0) 

    for _ in range(num_samples_per_class):
        data.append([np.random.uniform(180, 230), np.random.uniform(180, 230), np.random.uniform(50, 100), 
                     np.random.uniform(20, 40), np.random.uniform(100, 180), np.random.uniform(150, 220)])
        labels.append(1) 

    for _ in range(num_samples_per_class):
        data.append([np.random.uniform(50, 100), np.random.uniform(150, 200), np.random.uniform(50, 100), 
                     np.random.uniform(40, 90), np.random.uniform(80, 150), np.random.uniform(100, 200)])
        labels.append(2) 
        
    return np.array(data), np.array(labels), class_names

def load_real_images_and_labels(base_dir="../data_latih"): 
    data = []
    labels = []
    
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                if os.path.isfile(image_path) and not image_name.startswith('.'):
                    img = cv2.imread(image_path)
                    if img is not None:
                        full_mask = np.ones(img.shape[:2], dtype=np.bool_)
                        
                        features = extract_color_features_from_segmented_region(img, full_mask)
                        if features is not None:
                            data.append(features)
                            labels.append(class_to_idx[class_name])
    return np.array(data), np.array(labels), class_names

if __name__ == "__main__":
    use_dummy_dataset = False

    if use_dummy_dataset:
        X, y, class_names = create_dummy_dataset(num_samples_per_class=100)
    else:
        dataset_base_dir = "../data_latih" 
        try:
            X, y, class_names = load_real_images_and_labels(base_dir=dataset_base_dir)
            if len(X) == 0:
                X, y, class_names = create_dummy_dataset(num_samples_per_class=100)
                use_dummy_dataset = True
        except FileNotFoundError:
            X, y, class_names = create_dummy_dataset(num_samples_per_class=100)
            use_dummy_dataset = True

    if len(X) == 0:
        exit()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            exit()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_orange_red = np.array([0, 150, 100]) 
            upper_orange_red = np.array([25, 255, 255]) 

            lower_yellow_orange = np.array([20, 80, 80]) 
            upper_yellow_orange = np.array([40, 255, 255]) 

            lower_green_yellow = np.array([40, 50, 50])
            upper_green_yellow = np.array([90, 255, 255])

            mask_orange_red = cv2.inRange(frame_hsv, lower_orange_red, upper_orange_red)
            mask_yellow_orange = cv2.inRange(frame_hsv, lower_yellow_orange, upper_yellow_orange)
            mask_green_yellow = cv2.inRange(frame_hsv, lower_green_yellow, upper_green_yellow)
            
            combined_mask = cv2.bitwise_or(mask_orange_red, mask_yellow_orange)
            combined_mask = cv2.bitwise_or(combined_mask, mask_green_yellow)

            kernel = np.ones((5,5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            predicted_kematangan = "Tidak Terdeteksi"
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 2000: 
                    
                    final_segmented_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
                    cv2.drawContours(final_segmented_mask, [largest_contour], -1, 255, -1)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    current_features = extract_color_features_from_segmented_region(frame, final_segmented_mask)
                    
                    if current_features is not None:
                        predicted_label_index = knn_model.predict([current_features])[0]
                        predicted_kematangan = class_names[predicted_label_index]
                    else:
                        predicted_kematangan = "Tidak Cukup Data Jeruk Tersegmentasi"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Kematangan: {predicted_kematangan}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Mask Deteksi Jeruk', combined_mask)
            
            cv2.imshow('Deteksi Kematangan Jeruk Real-time', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if X.shape[1] >= 3:
            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
            plt.xlabel('Rata-rata Red')
            plt.ylabel('Rata-rata Green')
            plt.title('Sebaran Fitur Warna Jeruk (R vs G) dari Dataset')
            plt.colorbar(scatter, ticks=range(len(class_names)), label='Kematangan', format=plt.FuncFormatter(lambda i, *args: class_names[int(i)]))
            plt.grid(True)
            plt.show(block=False) 

        if X.shape[1] >= 6:
            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(X[:, 3], X[:, 4], c=y, cmap='viridis', s=50, alpha=0.8)
            plt.xlabel('Rata-rata Hue')
            plt.ylabel('Rata-rata Saturation')
            plt.title('Sebaran Fitur Warna Jeruk (H vs S) dari Dataset')
            plt.colorbar(scatter, ticks=range(len(class_names)), label='Kematangan', format=plt.FuncFormatter(lambda i, *args: class_names[int(i)]))
            plt.grid(True)
            plt.show(block=True)