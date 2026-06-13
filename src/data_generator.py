import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path

class RamDataGenerator(keras.utils.Sequence):
    """
    Tüm ham veriyi (örneğin 1 bearing için: ~2800 x 2560 x 2 matrisi) RAM'de tutar.
    Batch istendiğinde `seq_len` uzunluğundaki pencereleri anlık kesip döndürür.
    Bu sayede sliding window'ların gereksiz yere RAM'i 30 kat (seq_len kadar)
    şişirmesini önler, sadece birkaç yüz MB RAM harcar.
    """
    def __init__(
        self,
        X_base_list: list[np.ndarray],
        y_base_list: list[np.ndarray],
        seq_len: int = 30,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        X_base_list : [ (num_files_1, 2560, 2), (num_files_2, 2560, 2), ... ]
        y_base_list : [ (num_files_1,), (num_files_2,), ... ]
        """
        self.X_base_list = X_base_list
        self.y_base_list = y_base_list
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Geçerli (bearing_idx, start_idx) çiftlerini topla
        self.valid_indices = []
        for b_idx, X_base in enumerate(self.X_base_list):
            n_samples = len(X_base)
            if n_samples >= self.seq_len:
                for start_idx in range(n_samples - self.seq_len + 1):
                    self.valid_indices.append((b_idx, start_idx))
                    
        self.valid_indices = np.array(self.valid_indices)
        self.on_epoch_end()
        
    def __len__(self):
        if len(self.valid_indices) == 0:
            return 0
        return int(np.ceil(len(self.valid_indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_inds = self.valid_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Önceden bellek ayır (hızlı atama için)
        first_X = self.X_base_list[0]
        batch_X = np.empty((len(batch_inds), self.seq_len, first_X.shape[1], first_X.shape[2]), dtype=np.float32)
        batch_y = np.empty((len(batch_inds),), dtype=np.float32)
        
        for i, (b_idx, start_idx) in enumerate(batch_inds):
            X_b = self.X_base_list[b_idx]
            y_b = self.y_base_list[b_idx]
            
            end_idx = start_idx + self.seq_len
            batch_X[i] = X_b[start_idx : end_idx]
            batch_y[i] = y_b[end_idx - 1] # Hedef (target), pencerenin en son anındaki RUL değeridir
            
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle and len(self.valid_indices) > 0:
            np.random.shuffle(self.valid_indices)


def load_bearing_raw_data(bearing_dir: Path, rul_cap: float = 125.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Belirtilen klasördeki tüm acc_XXXXX.csv dosyalarını sırayla okur.
    Döndürülen:
        X_base: (num_files, 2560, 2)
        y_base: (num_files,)  (Health Indicator in [0, 1])
    Sadece 4. ve 5. sütunları (yatay ve dikey ivmelenme) alır.
    """
    files = sorted(list(bearing_dir.glob("acc_*.csv")))
    if not files:
        return np.empty((0, 2560, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    all_data = []
    num_files = len(files)
    
    for f in files:
        try:
            # Sadece ivmelenme kolonlarını al (sütun 4 ve 5)
            df = pd.read_csv(f, header=None, usecols=[4, 5])
            if len(df) == 2560:
                all_data.append(df.values.astype(np.float32))
            else:
                # Dosya bozuksa ya da eksikse 0 ile doldur
                arr = df.values.astype(np.float32)
                if len(arr) > 2560:
                    all_data.append(arr[:2560])
                else:
                    pad = np.zeros((2560 - len(arr), 2), dtype=np.float32)
                    all_data.append(np.vstack([pad, arr]))
        except Exception:
            # Okuma hatası olursa sıfır matrisi dön
            all_data.append(np.zeros((2560, 2), dtype=np.float32))
            
    X_base = np.array(all_data, dtype=np.float32)
    
    # RUL hesaplama: her dosya 10 saniye aralıklarla alınır.
    # Sondan başa doğru RUL artar.
    # index i için kalan süre (saniye) = (num_files - 1 - i) * 10
    rul_s = (num_files - 1 - np.arange(num_files)) * 10.0
    rul_min = rul_s / 60.0
    
    # 1. GÜNCELLEME: Hedefi Sağlık Yüzdesine (Health Indicator, 0.0 - 1.0) çevir
    # 125 dakika = 1.0 (tam sağlıklı), 0 dakika = 0.0 (arızalı)
    hi = rul_min / rul_cap
    y_base = np.clip(hi, 0.0, 1.0).astype(np.float32)
    
    return X_base, y_base
