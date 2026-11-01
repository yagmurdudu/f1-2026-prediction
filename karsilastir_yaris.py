"""
Meksika GP 2026 yarış sonucu ile yarıştan önce kaydedilen tahmini karşılaştırır.

Kullanım (yarıştan sonra):
    ./venv/bin/python karsilastir_yaris.py
        -> Sonucu Jolpica F1 API'den çeker.
    ./venv/bin/python karsilastir_yaris.py HAM RUS ANT ... (22 kod, 1.'den sonuncuya)
        -> API'de sonuç henüz yoksa sırayı elle ver. Bitiremeyenleri sona yaz ve başlarına ! koy (örn. !STR).

Çıktı: meksika_2026/yaris_gercek_karsilastirma.csv ve ekrana README'ye yapıştırılabilir tablo.
"""
import sys
import numpy as np, pandas as pd, requests
from scipy.stats import spearmanr

TAHMIN = "meksika_2026/yaris_tahmini.csv"
CIKTI = "meksika_2026/yaris_gercek_karsilastirma.csv"
KOD_DUZELT = {"HIR": "OCO"}  # Hirakawa, modelde Ocon'un koltuğu olarak yer alıyor

t = pd.read_csv(TAHMIN)

# ---- gerçek sonuç ----
if len(sys.argv) > 1:
    satir = []
    for i, k in enumerate(sys.argv[1:], 1):
        dnf = k.startswith("!")
        satir.append(dict(kod=KOD_DUZELT.get(k.lstrip("!").upper(), k.lstrip("!").upper()), gercek=i, bitirdi=not dnf))
    g = pd.DataFrame(satir)
    kaynak = "elle girildi"
else:
    d = requests.get("https://api.jolpi.ca/ergast/f1/2026/19/results.json", timeout=30).json()["MRData"]
    yaris = d["RaceTable"]["Races"]
    if not yaris:
        sys.exit("API'de Meksika 2026 yarış sonucu henüz yok. Sırayı elle ver: "
                 "./venv/bin/python karsilastir_yaris.py HAM RUS ... (bitiremeyenler !KOD)")
    g = pd.DataFrame([dict(kod=KOD_DUZELT.get(x["Driver"]["code"], x["Driver"]["code"]), gercek=int(x["position"]),
                           bitirdi=x["positionText"].isdigit(), durum=x["status"], puan=float(x["points"]))
                      for x in yaris[0]["Results"]])
    kaynak = "Jolpica F1 API"

eksik = set(t.kod) - set(g.kod)
if eksik:
    print("Uyarı: sonuçta olmayan pilotlar (karşılaştırmadan çıkarıldı):", eksik)
k = t.merge(g, on="kod", how="inner")
k["fark"] = k.tahmini_bitis - k.gercek
k = k.sort_values("gercek").reset_index(drop=True)

# ---- ölçüler ----
uyum = spearmanr(k.tahmini_bitis, k.gercek).statistic
hata = (k.tahmini_bitis.rank() - k.gercek.rank()).abs().mean()
grid_uyum = spearmanr(k.grid, k.gercek).statistic       # basit yöntem: "başladığı sırada bitirir"
grid_hata = (k.grid.rank() - k.gercek.rank()).abs().mean()
kazanan = k.loc[k.gercek == 1, "kod"].item()
tahmin_kazanan = t.loc[t.tahmini_bitis == 1, "kod"].item()
podyum_g = set(k[k.gercek <= 3].kod)
podyum_t = set(t[t.tahmini_bitis <= 3].kod)
puan_g = set(k[k.gercek <= 10].kod)
puan_t = set(t[t.tahmini_bitis <= 10].kod)
dnf_g = k[~k.bitirdi]
brier_kazanma = ((k.kazanma - (k.gercek == 1)) ** 2).sum()
ozet = {
    "Kaynak": kaynak,
    "Kazanan": f"{kazanan} (tahmin: {tahmin_kazanan}, kazanana verilen olasılık %{100 * k.loc[k.gercek == 1, 'kazanma'].item():.0f})",
    "Podyum isabeti": f"{len(podyum_g & podyum_t)}/3 (gerçek: {', '.join(k[k.gercek <= 3].kod)})",
    "Puan alan 10 pilot": f"{len(puan_g & puan_t)}/10",
    "Sıralama uyumu": f"{uyum:.3f} (grid sırası: {grid_uyum:.3f})",
    "Ortalama sıra hatası": f"{hata:.2f} (grid sırası: {grid_hata:.2f})",
    "Tam doğru / ±1 / ±2": f"{(k.fark == 0).sum()} / {(k.fark.abs() <= 1).sum()} / {(k.fark.abs() <= 2).sum()} (toplam {len(k)})",
    "Bitiremeyenler": (", ".join(f"{r.kod} (DNF olasılığı %{100 * r.dnf:.0f})" for r in dnf_g.itertuples()) or "yok"),
    "Ortalama DNF olasılığı": f"bitiremeyenlere %{100 * dnf_g.dnf.mean():.0f}, bitirenlere %{100 * k[k.bitirdi].dnf.mean():.0f}"
                              if len(dnf_g) else "-",
}
for a, b in ozet.items():
    print(f"{a:<24} {b}")

k[["gercek", "kod", "pilot", "grid", "tahmini_bitis", "fark", "bitirdi", "kazanma", "podyum", "dnf"]].to_csv(CIKTI, index=False)

# README için tablo
print("\n| Gerçek | Pilot | Başlangıç | Tahmin | Fark |\n|---|---|---|---|---|")
for r in k.itertuples():
    ad = r.pilot.replace("*", "") + ("" if r.bitirdi else " (DNF)")
    f = "✓" if r.fark == 0 else str(abs(int(r.fark)))
    print(f"| {r.gercek} | {ad} | {r.grid} | {r.tahmini_bitis} | {f} |")
# web sitesi de güncellensin
import os, shutil
if os.path.isdir("docs/data"):
    shutil.copyfile(CIKTI, "docs/data/meksika_yaris_karsilastirma.csv")
print(f"\nKaydedildi: {CIKTI} (ve docs/data/ içine kopyalandı)")
