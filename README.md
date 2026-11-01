# F1 2026 Şampiyonluk Tahmini

Makine öğrenmesiyle F1 2026 pilotlar ve takımlar şampiyonası, sıralama turu ve yarış tahmini.
**Web sitesi: https://yagmurdudu.github.io/f1-2026-prediction/**

## Gerçek bir yarış hafta sonunda deneme: Meksika GP 2026

Tahminler oturumlardan **önce** yapıldı ve kaydedildi, sonra gerçek sonuçla karşılaştırıldı.
Ayrıntılar: [`meksika_2026/`](meksika_2026/README.md)

### Sıralama turu: tahmin ve gerçek

| Gerçek | Pilot | Tahmin | Fark |
|---|---|---|---|
| 1 | Hamilton | 1 | ✓ |
| 2 | Russell | 3 | 1 |
| 3 | Leclerc | 4 | 1 |
| 4 | Verstappen | 2 | 2 |
| 5 | Antonelli | 5 | ✓ |
| 6 | Norris | 6 | ✓ |
| 7 | Hadjar | 8 | 1 |
| 8 | Lawson | 10 | 2 |
| 9 | Lindblad | 11 | 2 |
| 10 | Gasly | 9 | 1 |
| 11 | Hülkenberg | 12 | 1 |
| 12 | Colapinto | 13 | 1 |
| 13 | Piastri | 7 | 6 |
| 14 | Bortoleto | 14 | ✓ |
| 15 | Bearman | 15 | ✓ |
| 16 | Sainz | 17 | 1 |
| 17 | Albon | 20 | 3 |
| 18 | Alonso | 18 | ✓ |
| 19 | Hirakawa (Ocon yerine) | – | – |
| 20 | Stroll | 19 | 1 |
| 21 | Pérez | 22 | 1 |
| 22 | Bottas | 21 | 1 |

- **Pole doğru:** Hamilton.
- **Sıralama uyumu 0.96** (1 kusursuz). Ortalama sıra hatası 1,1. Q3'e kalan 10 pilotun 9'u doğru.
- 45 seanslık testlerde ortalama uyum 0.76; yani bu sonuç ortalamanın üstünde. Bir kısmı şans.

### Yarış: tahmin (1 Kasım 2026, 09:19 UTC'de, yarıştan önce kaydedildi)

Gerçek başlangıç sırasıyla 20.000 simülasyon. İlk 10:

| Tahmin | Pilot | Başlangıç | Kazanma | Podyum |
|---|---|---|---|---|
| 1 | Hamilton | 1 | %42 | %79 |
| 2 | Antonelli | 4 | %13 | %54 |
| 3 | Russell | 2 | %33 | %72 |
| 4 | Verstappen | 5 | %4 | %23 |
| 5 | Leclerc | 3 | %2 | %17 |
| 6 | Norris | 6 | %3 | %21 |
| 7 | Hadjar | 7 | %1 | %12 |
| 8 | Lindblad | 8 | <%1 | %7 |
| 9 | Piastri | 13 | <%1 | %6 |
| 10 | Lawson | 9 | <%1 | %3 |

Antonelli'nin bu yarışta şampiyonluğu garantileme olasılığı: **%95**.

Gerçek sonuçla karşılaştırma yarıştan sonra eklenecek (`karsilastir_yaris.py`).

## Kurulum (bir kez)

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Çalıştırma

Yeni bir yarış bittikten (ya da sıralama turu sonucu yayına girdikten) sonra:

```bash
./venv/bin/python veri_cek.py   # 2025–2026 sonuçlarını Jolpica F1 API'den indirir
./venv/bin/python model2.py     # Modeli eğitir, test eder, sezonu 20.000 kez simüle eder
```

Sadece model testini görmek için: `./venv/bin/python model2.py --sadece-test`

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `veri_cek.py` | Yarış ve sprint sonuçlarını, takvimi indirir → `data/sonuclar.csv`, `data/takvim.csv` |
| `model.py` | Model 1: form ve takım gücüyle tek aşamalı Random Forest (pist bilgisi yok) |
| `model2.py` | Model 2 (sayfada kullanılan): pist profili + önce sıralama turu, sonra yarış tahmini |
| `siralama_veri.py` | 2015–2026 tüm sıralama turu sonuçlarını indirir → `data/siralama.csv` |
| `meksika_siralama.py` | Meksika sıralama turu modeli (tur zamanı tabanlı, yarış modelinden bağımsız) |
| `model2.py --meksika-yaris` | Gerçek başlangıç sırasıyla Meksika yarış tahmini |
| `karsilastir_yaris.py` | Yarıştan sonra gerçek sonucu tahminle karşılaştırır |
| `meksika_2026/` | Meksika GP tahminleri, gerçek sonuçlar ve karşılaştırmalar |
| `data/m2_*.csv` | Model 2 çıktıları; sayfadaki her tablo ve grafik bunlardan |
| `data/*.csv` (m2_ olmayanlar) | Model 1 çıktıları |
| `docs/` | Web sitesi (GitHub Pages). Veriler `docs/data/` içinde |
| `site_verisi.py` | Model çıktılarını sitenin `docs/data/` klasörüne kopyalar |

## Model 2 kısaca

1. Her pistin 6 özelliği elle tanımlı (`PIST` sözlüğü): sokak, hız, aerodinamik yük, rakım, sollama, lastik aşınması.
2. Her yarış için yalnızca o yarıştan önceki verilerle form, takım gücü, yarış dışı kalma oranı ve benzer pistlerdeki performans hesaplanır.
3. İki Random Forest: biri başlangıç sırasını, diğeri başlangıç sırasından bitiriş sırasını tahmin eder. 2026 yarışları 3 kat ağırlıklı.
4. Kalan yarışlarda önce sıralama, sonra yarış rastgele oynatılır; rastgelelik payı modelin test hatasından alınır.
5. Şampiyonluğun kesinleşmesi, puan eşitliğinde galibiyet sayısı kuralını da hesaba katar.

## Test sonucu (2026'nın 13–18. yarışları, görmeden tahmin)

| Yöntem | Ortalama hata (sıra) | Sıralama uyumu |
|---|---|---|
| Son 6 yarış ortalaması | 3.26 | 0.709 |
| Model 1 | 3.29 | 0.730 |
| Model 2 (tahmini grid) | 3.35 | 0.701 |
| Model 2 (gerçek grid) | 3.21 | 0.744 |

Siteyi güncellemek için modelleri çalıştırdıktan sonra `./venv/bin/python site_verisi.py` çalıştırıp değişiklikleri GitHub'a gönder.

Siteyi yerelde açmak için:

```bash
python3 -m http.server 8000 -d docs
```

Sonra tarayıcıda http://localhost:8000 adresini aç.
