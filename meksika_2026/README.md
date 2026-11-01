# Meksika GP 2026 — sıralama ve yarış tahmini

Tur 19, Autódromo Hermanos Rodríguez. Sıralama turu 31 Ekim 2026'da yapıldı, yarış 1 Kasım 2026.

## 1. Sıralama turu tahmini (sıralamadan önce yapıldı)

Model: [`../meksika_siralama.py`](../meksika_siralama.py). Pole zamanına yüzde kaç uzak kalınacağını tahmin eden
doğrusal bir model kullanılıyor; her pilotun kendi tutarlılığına göre 20.000 kez simüle ediliyor.

### Gerçek sonuçla karşılaştırma

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
| 13 | **Piastri** | **7** | **6** |
| 14 | Bortoleto | 14 | ✓ |
| 15 | Bearman | 15 | ✓ |
| 16 | Sainz | 17 | 1 |
| 17 | Albon | 20 | 3 |
| 18 | Alonso | 18 | ✓ |
| 19 | Hirakawa (Ocon yerine) | – | – |
| 20 | Stroll | 19 | 1 |
| 21 | Pérez | 22 | 1 |
| 22 | Bottas | 21 | 1 |

- Sıralama uyumu: **0.96** (1 kusursuz; testlerde ortalama 0.76).
- Ortalama sıra hatası: **1,1**. 6 pilot tam doğru, 16 pilot en fazla 1 sıra farkla.
- Q3'e kalan 10 pilotun 9'u ve Q1'de elenen 6 pilotun 5'i doğru.
- En büyük yanılma: Piastri (7. tahmin, 13. oldu).

Gerçek sonuç kaynağı: [The Race](https://www.the-race.com/formula-1/f1-2026-mexican-grand-prix-qualifying-results/).
Diğer kaynaklarla (RacingNews365, GPFans, Autosport) karşılaştırıldı. Resmi F1 veri servisinde o sırada sonuç henüz yoktu.

## 2. Yarış tahmini (gerçek başlangıç sırasıyla)

Model: [`../model2.py`](../model2.py) içindeki yarış aşaması (`./venv/bin/python model2.py --meksika-yaris`).
Başlangıç sırası cezalar uygulanmış haliyle kullanıldı: Antonelli 4., Verstappen 5.; Lawson bir sıra geriden.
20.000 simülasyon yapıldı.

**Tahmin yarıştan önce kaydedildi: 1 Kasım 2026, 09:19 UTC** (yarış 20:00 UTC'de başlıyor).

Tahmini bitiş sırası, her simülasyondaki sıranın ortalamasına göre belirlendi; yarışı bitiremeyen pilot o simülasyonda en sona yazıldı.

| Tahmin | Pilot | Başlangıç | Kazanma | Podyum | Puan alma | Yarış dışı kalma | Beklenen puan |
|---|---|---|---|---|---|---|---|
| 1 | Hamilton | 1 | %42 | %79 | %93 | %7 | 18,1 |
| 2 | Antonelli | 4 | %13 | %54 | %95 | %4 | 13,6 |
| 3 | Russell | 2 | %33 | %72 | %89 | %11 | 16,5 |
| 4 | Verstappen | 5 | %4 | %23 | %80 | %14 | 8,4 |
| 5 | Leclerc | 3 | %2 | %17 | %80 | %11 | 7,5 |
| 6 | Norris | 6 | %3 | %21 | %76 | %18 | 7,9 |
| 7 | Hadjar | 7 | %1 | %12 | %75 | %12 | 6,2 |
| 8 | Lindblad | 8 | <%1 | %7 | %70 | %7 | 4,9 |
| 9 | Piastri | 13 | <%1 | %6 | %65 | %14 | 4,5 |
| 10 | Lawson | 9 | <%1 | %3 | %55 | %7 | 3,1 |
| 11 | Gasly | 10 | <%1 | %3 | %51 | %11 | 2,9 |
| 12 | Hülkenberg | 11 | <%1 | %2 | %45 | %18 | 2,3 |
| 13 | Colapinto | 12 | <%1 | %1 | %37 | %14 | 1,8 |
| 14 | Bortoleto | 14 | – | %1 | %31 | %11 | 1,3 |
| 15 | Bearman | 15 | <%1 | <%1 | %22 | %18 | 0,9 |
| 16 | Hirakawa (Ocon yerine) | 19 | – | <%1 | %15 | %7 | 0,5 |
| 17 | Sainz | 16 | – | <%1 | %8 | %11 | 0,3 |
| 18 | Alonso | 18 | – | – | %3 | %25 | 0,1 |
| 19 | Pérez | 21 | – | – | %3 | %22 | 0,1 |
| 20 | Albon | 17 | – | – | %3 | %25 | 0,1 |
| 21 | Bottas | 22 | – | – | %2 | %32 | 0,1 |
| 22 | Stroll | 20 | – | – | %2 | %46 | 0,1 |

Antonelli ortalamada 3,8. sırada bitiriyor ve yarışı en az bitirememe riski olan pilot, bu yüzden net sıralamada
2. sırada; ama kazanma olasılığı Russell'dan düşük (%13'e karşı %33).

### Yarıştan sonra karşılaştırma

```bash
./venv/bin/python karsilastir_yaris.py
```

Sonuç API'de yoksa sıra elle verilebilir; bitiremeyenlerin başına `!` konur:

```bash
./venv/bin/python karsilastir_yaris.py HAM RUS ANT ... '!STR'
```

Karşılaştırma `yaris_gercek_karsilastirma.csv` dosyasına yazılır. Modelin sonucu "herkes başladığı sırada bitirir"
basit yöntemiyle de karşılaştırılır.

**Şampiyonluk:** Antonelli'nin bugün şampiyonluğu garantileme olasılığı **%95**. İlk 9'da bitirirse kesin;
10. ya da daha geride kalırsa Russell'ın veya Hamilton'ın kazanmaması gerekiyor.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `siralama_tahmini.csv` | Pilot başına beklenen sıra, pole / ön sıra / ilk 3 / Q3 / Q1'de elenme olasılığı |
| `siralama_gercek_karsilastirma.csv` | Gerçek sıralama ile tahmin yan yana, başlangıç sırası dahil |
| `siralama_model_testi.csv` | Sıralama modelinin 45 seanslık testi (eski model ve basit yöntemle karşılaştırma) |
| `siralama_ozellik_onemi.csv` | Sıralama modelinin her bilgiye verdiği ağırlık |
| `gecmis_meksika_siralamalari.csv` | Bugünkü pilotların 2015–2025 Meksika sıralama sonuçları |
| `yaris_tahmini.csv` | Yarış tahmini: net bitiş sırası, kazanma, podyum, puan alma, yarış dışı kalma, beklenen puan, sıralama öncesi kazanma tahmini ve tahminin kaydedildiği zaman |
| `yaris_gercek_karsilastirma.csv` | (yarıştan sonra oluşacak) gerçek sonuç ile tahmin yan yana |

## Sınırlamalar

- Yarış modeli test yarışlarında ortalamada yaklaşık 3 sıra yanılıyor. Strateji, safety car ve ilk viraj olaylarını bilmiyor.
- Hirakawa'nın koltuğu, modelde Ocon'un verileriyle temsil ediliyor.
- Kaynaklar cezayı kimin aldığı konusunda çelişiyor, ama başlangıç sırası (Antonelli 4., Verstappen 5.) konusunda hemfikir.
