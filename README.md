
### SERVER UBUNTU
```txt

```
### GITHUB

```txt
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/edycoleee/cnn2025.git
git push -u origin main
```

### REQUIREMENT
```py
# 1. Membuat Virtual Environtment
python3 -m venv venv
source venv/bin/activate  #Linux / Macbook
venv\Scripts\activate # Windows

#2. Install library
pip install matplotlib numpy
pip install seaborn
pip install scikit-learn
pip install tensorflow
pip install flask
```

# RAPBERYY PI 5

```py
sudo apt update
sudo apt upgrade -y

sudo apt install -y build-essential libssl-dev zlib1g-dev \
libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
tk-dev libffi-dev wget

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
sudo tar xzf Python-3.10.14.tgz
cd Python-3.10.14
sudo ./configure --enable-optimizations
sudo make -j4
sudo make altinstall

python3.10 --version

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install tensorflow==2.13.0


python3 -c "import tensorflow as tf; print(tf.__version__)"
```

# GPU RTX 2060 Anda di laptop Asus ROG

install python versi 3.10

```
Cara Menginstal cuDNN (Zip Method)
Asumsi: Anda telah menginstal CUDA Toolkit 11.2 di lokasi default Windows, yaitu:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\
Langkah 1: Ekstrak File ZIP cuDNN
Temukan file ZIP cuDNN yang baru saja Anda unduh (misalnya, cudnn-windows-x86_64-8.1.0.27_cuda11.2-archive.zip).
Klik kanan pada file ZIP tersebut, lalu pilih Extract All... atau gunakan software seperti 7-Zip/WinRAR untuk mengekstrak isinya.
Setelah diekstrak, Anda akan mendapatkan sebuah folder baru dengan nama yang sama, dan di dalamnya terdapat tiga sub-folder utama:
- bin
- include
- lib
Langkah 2: Salin File ke Direktori CUDA
Sekarang, Anda perlu menyalin konten dari folder yang diekstrak tadi ke folder instalasi CUDA Anda.
Buka folder hasil ekstraksi cuDNN tadi.
Buka jendela File Explorer baru, dan navigasikan ke lokasi instalasi CUDA:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\
Salin konten:
Buka folder bin di folder cuDNN yang diekstrak. Salin semua file DLL di dalamnya.
Rekatkan (Paste) file-file tersebut ke dalam folder bin di direktori CUDA (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin).
Lakukan hal yang sama untuk folder include dan lib.
Singkatnya, Anda memastikan bahwa file-file berikut berada di tempat yang benar:
File cuDNN	Lokasi Tujuan
...ekstrak...\bin\*.*	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
...ekstrak...\include\*.*	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include
...ekstrak...\lib\*.*	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib
```
```cmd
py -3.10 --version

py -3.10 -m venv tf_gpu_env

.\tf_gpu_env\Scripts\activate
python --version 

python.exe -m pip install --upgrade pip
pip install tensorflow==2.10.0
pip install "numpy<2"
```
```
1. Verifikasi Ulang Pengaturan Path
Tekan tombol Windows + S, ketik "environment variables", dan buka "Edit the system environment variables".
Klik Environment Variables....
Di bagian System variables (bukan User variables), cari dan pilih Path, lalu klik Edit....
Pastikan dua jalur (path) berikut persis ada di daftar tersebut:

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

Restart Komputer Anda
```

```py
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

```

```py
import tensorflow as tf

# Pastikan TensorFlow menggunakan GPU secara default
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please check your GPU setup, the default device is not set.")

# Contoh sederhana operasi di GPU
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([4.0, 5.0, 6.0], shape=[3], name='b')
    c = a + b

print(c)
```

🧩 Tahapan Dasar CNN (dihubungkan ke script)
1. Input Data (Gambar)
Script: Membuat gambar 32×32, separuh atas hitam, separuh bawah putih.

Konsep CNN: Input adalah matriks piksel (grayscale atau RGB).

Yang perlu dipelajari:

Representasi gambar sebagai array (nilai intensitas piksel).

Normalisasi data (misalnya skala 0–1).

Perbedaan grayscale vs RGB (channel).

Bentuk input CNN: (height, width, channels).

2. Kernel / Filter
Script: Kernel acak 3×3.

Konsep CNN: Filter kecil yang bergerak di atas gambar untuk mengekstrak fitur (tepi, tekstur, pola).

Yang perlu dipelajari:

Operasi konvolusi (matriks kernel digeser di atas gambar).

Peran kernel dalam mendeteksi pola (horizontal edge, vertical edge, dll).

Padding, stride, dan efek ukuran kernel.

Hubungan kernel dengan feature map.

3. Convolution Operation
Script: cv2.filter2D(img, -1, kernel)

Konsep CNN: Menghasilkan feature map dari input dengan kernel.

Yang perlu dipelajari:

Bagaimana konvolusi bekerja secara matematis.

Feature map sebagai representasi fitur yang terdeteksi.

Perbedaan convolution vs correlation.

Multi-channel convolution (untuk RGB).

4. Prediction & Loss Function
Script: predict() → rata-rata feature map sebagai skor, lalu loss_fn() → MSE.

Konsep CNN: CNN menghasilkan prediksi, lalu dibandingkan dengan target menggunakan fungsi loss.

Yang perlu dipelajari:

Fungsi loss umum: MSE, cross-entropy.

Bagaimana loss mengukur “seberapa jauh” prediksi dari target.

Peran loss dalam mengarahkan training.

5. Backpropagation & Gradient
Script: compute_grad() dengan finite difference.

Konsep CNN: Gradien menunjukkan arah perubahan kernel agar loss menurun.

Yang perlu dipelajari:

Konsep turunan (derivative) dalam optimisasi.

Backpropagation di CNN (chain rule).

Numerical gradient vs analytical gradient.

Bagaimana gradien memodifikasi kernel.

6. Training Loop
Script: 20 iterasi update kernel dengan kernel -= lr * grad.

Konsep CNN: Kernel diperbarui sedikit demi sedikit agar lebih baik mendeteksi pola.

Yang perlu dipelajari:

Optimizer (SGD, Adam, RMSProp).

Learning rate dan dampaknya.

Epoch, batch size, iterasi.

Overfitting vs generalisasi.

7. Visualisasi Kernel & Loss
Script: Plot kernel awal, tengah, akhir + plot loss.

Konsep CNN: Visualisasi membantu memahami bagaimana kernel belajar dan loss menurun.

Yang perlu dipelajari:

Interpretasi kernel (misalnya kernel edge detector).

Loss curve: apakah menurun stabil atau tidak.

Debugging training dengan visualisasi.

====================================================================

