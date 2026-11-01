"""Web sitesinin (docs/) kullandığı CSV dosyalarını docs/data/ içine kopyalar.
Modeller yeniden çalıştırıldıktan sonra:  ./venv/bin/python site_verisi.py"""
import os, shutil

DOSYALAR = {
    # sitedeki ad           : kaynak
    "m2_tahmin.csv":          "data/m2_tahmin.csv",
    "m2_kalan.csv":           "data/m2_kalan.csv",
    "m2_dogrulama.csv":       "data/m2_dogrulama.csv",
    "m2_onem.csv":            "data/m2_onem.csv",
    "m2_yarislar.csv":        "data/m2_yarislar.csv",
    "m2_pistler.csv":         "data/m2_pistler.csv",
    "m2_ilerleme.csv":        "data/m2_ilerleme.csv",
    "m2_takimlar.csv":        "data/m2_takimlar.csv",
    "m2_dagilim.csv":         "data/m2_dagilim.csv",
    "m2_kesinlesme.csv":      "data/m2_kesinlesme.csv",
    "m2_meksika.csv":         "data/m2_meksika.csv",
    "meksika_siralama_karsilastirma.csv": "meksika_2026/siralama_gercek_karsilastirma.csv",
    "meksika_yaris_tahmini.csv":          "meksika_2026/yaris_tahmini.csv",
    "meksika_yaris_karsilastirma.csv":    "meksika_2026/yaris_gercek_karsilastirma.csv",  # yarıştan sonra oluşur
}

os.makedirs("docs/data", exist_ok=True)
for hedef, kaynak in DOSYALAR.items():
    if os.path.exists(kaynak):
        shutil.copyfile(kaynak, f"docs/data/{hedef}")
        print("kopyalandı:", hedef)
    else:
        print("yok (atlandı):", kaynak)
