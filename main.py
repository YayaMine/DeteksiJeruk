import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

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

    avg_b = np.mean(masked_bgr_pixels[:, 0])
    avg_g = np.mean(masked_bgr_pixels[:, 1])
    avg_r = np.mean(masked_bgr_pixels[:, 2])

    masked_hsv_pixels = cv2.cvtColor(masked_bgr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    avg_h = np.mean(masked_hsv_pixels[:, 0])
    avg_s = np.mean(masked_hsv_pixels[:, 1])
    avg_v = np.mean(masked_hsv_pixels[:, 2])

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

def load_real_images_and_labels(base_dir="data_latih"):
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

def predict_orange_ripeness(image_path, knn_model, class_names):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not load image from path."

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange_red = np.array([0, 150, 100])
    upper_orange_red = np.array([25, 255, 255])

    lower_yellow_orange = np.array([20, 80, 80])
    upper_yellow_orange = np.array([40, 255, 255])

    lower_green_yellow = np.array([40, 50, 50])
    upper_green_yellow = np.array([90, 255, 255])

    mask_orange_red = cv2.inRange(img_hsv, lower_orange_red, upper_orange_red)
    mask_yellow_orange = cv2.inRange(img_hsv, lower_yellow_orange, upper_yellow_orange)
    mask_green_yellow = cv2.inRange(img_hsv, lower_green_yellow, upper_green_yellow)
    
    combined_mask = cv2.bitwise_or(mask_orange_red, mask_yellow_orange)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green_yellow)

    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_kematangan = "Tidak Terdeteksi"
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000:
            final_segmented_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
            cv2.drawContours(final_segmented_mask, [largest_contour], -1, 255, -1)

            current_features = extract_color_features_from_segmented_region(img, final_segmented_mask > 0)
            
            if current_features is not None:
                predicted_label_index = knn_model.predict([current_features])[0]
                predicted_kematangan = class_names[predicted_label_index]
            else:
                predicted_kematangan = "Tidak Cukup Data Jeruk Tersegmentasi"
        else:
            predicted_kematangan = "Objek terlalu kecil atau tidak terdeteksi sebagai jeruk."
    else:
        predicted_kematangan = "Tidak ada objek jeruk yang terdeteksi dalam gambar."
    
    return predicted_kematangan

if __name__ == "__main__":
    use_dummy_dataset = False

    print("Memuat dataset untuk pelatihan model...")
    if use_dummy_dataset:
        X, y, class_names = create_dummy_dataset(num_samples_per_class=100)
        print("Menggunakan dataset dummy.")
    else:
        dataset_base_dir = "../data_latih"
        try:
            X, y, class_names = load_real_images_and_labels(base_dir=dataset_base_dir)
            if len(X) == 0:
                print(f"Tidak ada gambar ditemukan di '{dataset_base_dir}'. Menggunakan dataset dummy.")
                X, y, class_names = create_dummy_dataset(num_samples_per_class=100)
                use_dummy_dataset = True
            else:
                print(f"Berhasil memuat {len(X)} sampel dari '{dataset_base_dir}'.")
        except FileNotFoundError:
            print(f"Direktori '{dataset_base_dir}' tidak ditemukan. Menggunakan dataset dummy.")
            X, y, class_names = create_dummy_dataset(num_samples_per_class=100)
            use_dummy_dataset = True

    if len(X) == 0:
        print("Tidak ada data untuk melatih model. Keluar.")
        exit()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel dilatih dengan akurasi: {accuracy:.2f}")

    print("\n--- Deteksi Kematangan Jeruk dari File Gambar ---")

    root = tk.Tk()
    root.withdraw()

    while True:
        messagebox.showinfo("Pilih Gambar", "Silakan pilih file gambar jeruk untuk dideteksi.")
        
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Jeruk",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )

        if not file_path:
            print("Pemilihan gambar dibatalkan. Keluar dari program.")
            break
        
        print(f"Memproses gambar: {file_path}")
        predicted_ripeness = predict_orange_ripeness(file_path, knn_model, class_names)
        
        print(f"Kematangan jeruk pada '{os.path.basename(file_path)}': {predicted_ripeness}")
        print("-" * 50)

        if not messagebox.askyesno("Deteksi Selesai", "Kematangan: " + predicted_ripeness + "\n\nDeteksi gambar lain?"):
            break

    root.destroy()
    print("Program selesai.")

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