"""
F1 2026 şampiyonluk tahmini — model 2 (pistleri ayırt eden, iki aşamalı)

Aşama 1 (sıralama turları): pilotun başlangıç sırasını (grid) tahmin eder.
Aşama 2 (yarış): grid + form + takım gücü + "bu tür pistte nasıl" bilgisinden bitiriş sırasını tahmin eder.
Pist profili: her pist elle tanımlanmış özelliklerle (sokak, hız, yük, yükseklik, sollama, lastik aşınması)
anlatılır. Pilot/takımın 2026'da BENZER pistlerde kendi ortalamasına göre ne kadar iyi olduğu hesaplanır.
Simülasyon: önce grid, sonra yarış rastgele çekilir; 20.000 sezon.
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

PUAN = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
N_SIM = 20000
rng = np.random.default_rng(42)

# ---- Pist profilleri (genel bilinen karakteristikler, 1 = düşük, 3 = yüksek) ----
# sokak: 0 kalıcı pist, 0.5 yarı sokak, 1 sokak | hiz: düzlük/en yüksek hız | yuk: aerodinamik yük ihtiyacı
# yukseklik: km | sollama: sollama kolaylığı | asinma: lastik aşınması
PIST = {
    #               sokak hiz yuk yukseklik sollama asinma
    'albert_park':  (0.5, 2, 2, 0.0, 2, 2), 'shanghai':     (0, 2, 2, 0.0, 3, 3),
    'suzuka':       (0, 2, 3, 0.0, 1, 3),   'bahrain':      (0, 2, 2, 0.0, 3, 3),
    'jeddah':       (1, 3, 1, 0.0, 2, 1),   'miami':        (0.5, 2, 2, 0.0, 2, 2),
    'imola':        (0, 2, 2, 0.0, 1, 2),   'monaco':       (1, 1, 3, 0.0, 1, 1),
    'catalunya':    (0, 2, 3, 0.0, 2, 3),   'villeneuve':   (0.5, 3, 1, 0.0, 3, 2),
    'red_bull_ring':(0, 3, 2, 0.7, 3, 2),   'silverstone':  (0, 3, 3, 0.0, 2, 3),
    'spa':          (0, 3, 1, 0.4, 3, 2),   'hungaroring':  (0, 1, 3, 0.0, 1, 2),
    'zandvoort':    (0, 1, 3, 0.0, 1, 2),   'monza':        (0, 3, 1, 0.0, 3, 1),
    'madring':      (0.5, 2, 2, 0.6, 2, 2), 'baku':         (1, 3, 1, 0.0, 3, 1),
    'sepang':       (0, 2, 2, 0.0, 3, 3),   'marina_bay':   (1, 1, 3, 0.0, 1, 2),
    'americas':     (0, 2, 2, 0.0, 2, 3),   'rodriguez':    (0, 2, 3, 2.2, 2, 1),
    'interlagos':   (0, 2, 2, 0.8, 3, 2),   'vegas':        (1, 3, 1, 0.6, 3, 1),
    'losail':       (0, 2, 3, 0.0, 2, 3),   'yas_marina':   (0, 2, 2, 0.0, 2, 2),
}
PCOLS = ['p_sokak', 'p_hiz', 'p_yuk', 'p_yukseklik', 'p_sollama', 'p_asinma']
_P = pd.DataFrame(PIST, index=PCOLS).T
_PZ = (_P - _P.mean()) / _P.std()  # benzerlik için standartlaştırılmış


def benzerlik(c1, c2):
    if c1 not in _PZ.index or c2 not in _PZ.index:
        return 0.0
    d2 = float(((_PZ.loc[c1] - _PZ.loc[c2]) ** 2).sum())
    return np.exp(-d2 / 4)


res = pd.read_csv("data/sonuclar.csv")
cal = pd.read_csv("data/takvim.csv")
races = res[res.kind == "race"].merge(cal[["season", "round", "circuit"]], on=["season", "round"])
races["dnf"] = (~races.classified).astype(int)
races["grid"] = races.grid.replace(0, 22)  # pit çıkışından başlayan
races["idx"] = races.season * 100 + races["round"]
order = sorted(races.idx.unique())


def ozellikler(gecmis, pilotlar, sezon, circuit):
    bu = gecmis[gecmis.season == sezon]
    onceki = gecmis[gecmis.season == sezon - 1]
    son = bu if len(bu) else onceki
    son5 = son[son.idx.isin(sorted(son.idx.unique())[-5:])]
    takim_p5 = son5.groupby("team").points.mean()
    takim_g5 = son5.groupby("team").grid.mean()
    takim_ort = son.groupby("team").pos.mean()
    # benzer pist ağırlıkları (bu sezonun yarışları; sezon başında geçen sezon)
    w = son.drop_duplicates("idx").set_index("idx").circuit.map(lambda c: benzerlik(c, circuit))
    son = son.assign(w=son.idx.map(w))
    # her yarışta pilotun/takımın kendi ortalamasından sapması
    son = son.assign(p_sapma=son.pos - son.groupby("driver").pos.transform("mean"),
                     g_sapma=son.grid - son.groupby("driver").grid.transform("mean"),
                     t_sapma=son.pos - son.groupby("team").pos.transform("mean"))
    prof = PIST.get(circuit, (np.nan,) * 6)
    rows = []
    for d, t in pilotlar:
        g = son[son.driver == d].sort_values("idx")
        tg = son[son.team == t]
        p = onceki[(onceki.driver == d) & (onceki.circuit == circuit)]
        wsum, twsum = g.w.sum(), tg.w.sum()
        rows.append(dict(
            driver=d, team=t,
            form3=g.pos.tail(3).mean(), form6=g.pos.tail(6).mean(), puan5=g.points.tail(5).mean(),
            sezon_ort=g.pos.mean(), grid5=g.grid.tail(5).mean(), grid_ort=g.grid.mean(),
            takim_puan5=takim_p5.get(t, np.nan), takim_grid5=takim_g5.get(t, np.nan), takim_ort=takim_ort.get(t, np.nan),
            dnf_orani=(g.dnf.sum() + 1) / (len(g) + 10),
            pist_gecmis=p.pos.mean() if len(p) else np.nan,
            benzer_pilot=(g.p_sapma * g.w).sum() / wsum if wsum > 0.3 else np.nan,
            benzer_grid=(g.g_sapma * g.w).sum() / wsum if wsum > 0.3 else np.nan,
            benzer_takim=(tg.t_sapma * tg.w).sum() / twsum if twsum > 0.3 else np.nan,
            yaris_no=len(bu.idx.unique()),
            **dict(zip(PCOLS, prof)),
        ))
    return pd.DataFrame(rows)


# ---- eğitim seti ----
X_list = []
for ix in order[1:]:
    hedef = races[races.idx == ix]
    gecmis = races[races.idx < ix]
    f = ozellikler(gecmis, list(zip(hedef.driver, hedef.team)), hedef.season.iloc[0], hedef.circuit.iloc[0])
    f["pos"], f["grid"], f["idx"] = hedef.pos.values, hedef.grid.values, ix
    X_list.append(f)
data = pd.concat(X_list, ignore_index=True)

ESKI = ["form3", "form6", "puan5", "sezon_ort", "takim_puan5", "takim_ort", "dnf_orani", "pist_gecmis", "yaris_no"]
Q_FEATS = ["grid5", "grid_ort", "takim_grid5", "form6", "takim_puan5", "benzer_grid", "benzer_takim", "yaris_no"] + PCOLS
R_FEATS = ["grid", "form3", "form6", "puan5", "sezon_ort", "takim_puan5", "takim_ort", "dnf_orani",
           "benzer_pilot", "benzer_takim", "yaris_no"] + PCOLS
MED = {c: data[c].median() for c in set(Q_FEATS + R_FEATS + ESKI) if c != "grid"}
for c, v in MED.items():
    data[c] = data[c].fillna(v)


def rf(leaf=20):
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, random_state=0, n_jobs=-1)


def agirlik(df):
    return np.where(df.idx > 202600, 3.0, 1.0)


# ---- doğrulama: 2026 R13-R18 ----
test_ix = [ix for ix in order if ix >= 202613]
tr, te = data[~data.idx.isin(test_ix)], data[data.idx.isin(test_ix)]


def spear(p):
    return float(np.mean([spearmanr(p[(te.idx == ix).values], te.pos[te.idx == ix]).statistic for ix in test_ix]))


m_eski = rf().fit(tr[ESKI], tr.pos, sample_weight=agirlik(tr))
p_eski = m_eski.predict(te[ESKI])
qm = rf().fit(tr[Q_FEATS], tr.grid, sample_weight=agirlik(tr))
rm = rf().fit(tr[R_FEATS], tr.pos, sample_weight=agirlik(tr))
g_tahmin = qm.predict(te[Q_FEATS])
p_yeni = rm.predict(te[R_FEATS].assign(grid=g_tahmin))           # grid bilinmiyor (gelecek yarışlar)
p_grid = rm.predict(te[R_FEATS])                                   # gerçek grid biliniyor (sıralama sonrası)
dogrulama = [
    dict(yontem="Son 6 yarış ortalaması", mae=mean_absolute_error(te.pos, te.form6), spearman=spear(te.form6.values)),
    dict(yontem="Model 1 (pist bilgisi yok)", mae=mean_absolute_error(te.pos, p_eski), spearman=spear(p_eski)),
    dict(yontem="Model 2 (pist profili + tahmini grid)", mae=mean_absolute_error(te.pos, p_yeni), spearman=spear(p_yeni)),
    dict(yontem="Model 2, gerçek grid ile", mae=mean_absolute_error(te.pos, p_grid), spearman=spear(p_grid)),
    dict(yontem="Model 1 + Model 2 ortalaması", mae=mean_absolute_error(te.pos, (p_eski + p_yeni) / 2),
         spearman=spear((p_eski + p_yeni) / 2)),
]
import sys
if "--sadece-test" in sys.argv:
    for r in dogrulama:
        print(f"{r['yontem']:<42} hata {r['mae']:.3f}  uyum {r['spearman']:.3f}")
    sys.exit()
q_dog = dict(mae=mean_absolute_error(te.grid, g_tahmin),
             spearman=float(np.mean([spearmanr(g_tahmin[(te.idx == ix).values], te.grid[te.idx == ix]).statistic for ix in test_ix])))
for r in dogrulama:
    print(f"{r['yontem']:<42} hata {r['mae']:.3f}  uyum {r['spearman']:.3f}")
print(f"{'Sıralama turu tahmini':<42} hata {q_dog['mae']:.3f}  uyum {q_dog['spearman']:.3f}")

# belirsizlik payları: doğrulamadaki hatalardan (DNF'ler hariç)
bit = (te.pos <= 18).values
q_std = float(np.std(te.grid.values - g_tahmin))
r_std = float(np.std(te.pos.values[bit] - p_grid[bit]))
print("grid std", round(q_std, 2), "yarış std", round(r_std, 2))

# ---- final modeller: tüm veri ----
qm = rf().fit(data[Q_FEATS], data.grid, sample_weight=agirlik(data))
rm = rf().fit(data[R_FEATS], data.pos, sample_weight=agirlik(data))
onem_q = dict(zip(Q_FEATS, qm.feature_importances_))
onem_r = dict(zip(R_FEATS, rm.feature_importances_))

# ---- kalan yarışlar ----
son_yaris = races[races.idx == order[-1]]
pilotlar = list(zip(son_yaris.driver, son_yaris.team))
d_ids = [d for d, _ in pilotlar]
n = len(d_ids)
takim_of = dict(pilotlar)
kodlar = dict(zip(res.driver, res.code))
soyad = lambda d: d.split('_')[-1].capitalize()
kalan = cal[(cal.season == 2026) & (cal["round"] > races[races.season == 2026]["round"].max())].reset_index(drop=True)
s26 = res[res.season == 2026]
mevcut = s26.groupby("driver").points.sum()
galibiyet = s26[(s26.kind == "race") & (s26.pos == 1)].driver.value_counts()

# ---- Meksika 2026: GERÇEK başlangıç sırasıyla yarış tahmini ----
# Kullanım: ./venv/bin/python model2.py --meksika-yaris
# Sıralama sonucu (The Race) + cezalar: Verstappen/Antonelli yer değişti (Antonelli 4., Verstappen 5.),
# Lawson bir sıra geriye. Haas'ta Ocon yerine Hirakawa yarışıyor: modelde o koltuk Ocon'un verileriyle temsil ediliyor.
if "--meksika-yaris" in sys.argv:
    GRID_MEX = ['hamilton', 'russell', 'leclerc', 'antonelli', 'max_verstappen', 'norris', 'hadjar', 'arvid_lindblad',
                'lawson', 'gasly', 'hulkenberg', 'colapinto', 'piastri', 'bortoleto', 'bearman', 'sainz', 'albon',
                'alonso', 'ocon', 'stroll', 'perez', 'bottas']
    eksik = set(GRID_MEX) - set(d_ids)
    if eksik:  # sürücü kimlikleri API'deki ile eşleşmeli
        GRID_MEX = [next((d for d in d_ids if d.endswith(g.split('_')[-1])), g) for g in GRID_MEX]
    assert set(GRID_MEX) == set(d_ids), set(GRID_MEX) ^ set(d_ids)
    grid_of = {d: i + 1 for i, d in enumerate(GRID_MEX)}
    f = ozellikler(races, pilotlar, 2026, 'rodriguez')
    for c, v in MED.items():
        if c in f:
            f[c] = f[c].fillna(v)
    f['grid'] = f.driver.map(grid_of)
    bek = rm.predict(f[R_FEATS])
    dnf_p = f.dnf_orani.values
    puan_tablo = np.array([PUAN.get(k, 0) for k in range(1, n + 1)])
    skor = bek + rng.normal(0, r_std, (N_SIM, n))
    dnf = rng.random((N_SIM, n)) < dnf_p
    skor[dnf] += 100
    sira = skor.argsort(axis=1).argsort(axis=1)
    sira[dnf] = 99
    puan = np.where(sira < n, puan_tablo[np.minimum(sira, n - 1)], 0)
    # şampiyonluk bugün kesinleşir mi? (fark > 4×25, ya da tam 100 ve galibiyet sayısı Antonelli lehine)
    T = np.array([mevcut.get(d, 0) for d in d_ids])[None, :] + puan
    W = np.array([galibiyet.get(d, 0) for d in d_ids])[None, :] + (sira == 0)
    ant = d_ids.index('antonelli')
    kalan_sonra = 4
    fark = T[:, [ant]] - T
    ok = (fark > 25 * kalan_sonra) | ((fark == 25 * kalan_sonra) & (W[:, [ant]] > W + kalan_sonra))
    ok[:, ant] = True
    kesin = ok.all(axis=1)
    m_tum = pd.read_csv('data/m2_yarislar.csv').query('tur == 19').set_index('kod')
    out = pd.DataFrame(dict(
        kod=[kodlar[d] for d in d_ids], pilot=[soyad(d) for d in d_ids], takim=[takim_of[d] for d in d_ids],
        grid=[grid_of[d] for d in d_ids],
        beklenen_sira=np.nanmean(np.where(sira < 99, sira + 1, np.nan), axis=0).round(1),  # bitirdiği yarışlarda
        kazanma=(sira == 0).mean(axis=0).round(4), podyum=(sira < 3).mean(axis=0).round(4),
        puan_alma=(sira < 10).mean(axis=0).round(4), dnf=(sira == 99).mean(axis=0).round(4),
        beklenen_puan=puan.mean(axis=0).round(2),
    ))
    out['kazanma_gridsiz'] = out.kod.map(m_tum.kazanma)  # sıralama bilinmeden önceki tahmin
    # Net bitiş sırası tahmini: her simülasyonda yarış dışı kalan en sona yazılır, ortalamaya göre sıralanır
    out['ort_sira_dnf_dahil'] = np.where(sira < 99, sira + 1, n).mean(axis=0).round(2)
    out = out.sort_values(['ort_sira_dnf_dahil', 'beklenen_puan'], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, 'tahmini_bitis', range(1, n + 1))
    out['tahmin_zamani_utc'] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')
    out.loc[out.pilot == 'Ocon', 'pilot'] = 'Hirakawa*'
    out.to_csv('data/meksika_yaris_tahmin.csv', index=False)
    pd.set_option('display.width', 200)
    print(out.to_string(index=False))
    # Antonelli'nin sırasına göre
    a = sira[:, ant]
    print(f"\nAntonelli'nin bugün şampiyonluğu garantileme olasılığı: {kesin.mean():.4f}")
    print("Antonelli'nin bitirme dağılımı:", {f'P{k+1}': round(float((a == k).mean()), 3) for k in range(10)},
          "puan dışı/DNF:", round(float((a >= 10).mean()), 3))
    print(f"Yarış belirsizlik payı (test hatası): ±{r_std:.2f} sıra")
    sys.exit()

# Her yarış: grid tahmini + her olası grid (1..n) için yarış tahmini (simülasyonda tablo olarak kullanılır)
g_bek, r_tablo, dnf_p, yaris_ozellik = [], [], None, []
for _, r in kalan.iterrows():
    f = ozellikler(races, pilotlar, 2026, r.circuit)
    for c, v in MED.items():
        if c in f:
            f[c] = f[c].fillna(v)
    g_bek.append(qm.predict(f[Q_FEATS]))
    tab = np.zeros((n, n))
    for g in range(1, n + 1):
        tab[:, g - 1] = rm.predict(f[R_FEATS[1:]].assign(grid=g)[R_FEATS])
    r_tablo.append(tab)
    dnf_p = f.dnf_orani.values
    yaris_ozellik.append(f[["driver", "benzer_pilot", "benzer_takim", "benzer_grid"]].assign(tur=int(r["round"])))
g_bek = np.array(g_bek)

puan_tablo = np.array([PUAN.get(k, 0) for k in range(1, n + 1)])
toplam = np.tile([mevcut.get(d, 0) for d in d_ids], (N_SIM, 1)).astype(float)
wins = np.tile([galibiyet.get(d, 0) for d in d_ids], (N_SIM, 1)).astype(float)
yaris_sira, yaris_grid, ara_toplam, ara_wins = [], [], [], []
satir_idx = np.arange(n)
for k in range(len(kalan)):
    # 1) sıralama turu
    gs = g_bek[k] + rng.normal(0, q_std, (N_SIM, n))
    grid = gs.argsort(axis=1).argsort(axis=1)  # 0 = pole
    # 2) yarış: o gridden beklenen sıra + gürültü
    bek = r_tablo[k][satir_idx[None, :], grid]
    skor = bek + rng.normal(0, r_std, (N_SIM, n))
    dnf = rng.random((N_SIM, n)) < dnf_p
    skor[dnf] += 100
    sira = skor.argsort(axis=1).argsort(axis=1)
    sira[dnf] = 99
    toplam += np.where(sira < n, puan_tablo[np.minimum(sira, n - 1)], 0)
    wins += (sira == 0)
    yaris_sira.append(sira.copy()); yaris_grid.append(grid); ara_toplam.append(toplam.copy()); ara_wins.append(wins.copy())

final_sira = (-(toplam + wins * 1e-3)).argsort(axis=1).argsort(axis=1)
sira_ort = np.argsort(-toplam.mean(axis=0))
ilk = [d_ids[j] for j in sira_ort[:8]]

TAKIM = {'mercedes': 'Mercedes', 'ferrari': 'Ferrari', 'red_bull': 'Red Bull', 'mclaren': 'McLaren',
         'rb': 'Racing Bulls', 'alpine': 'Alpine', 'williams': 'Williams', 'haas': 'Haas', 'audi': 'Audi',
         'cadillac': 'Cadillac', 'aston_martin': 'Aston Martin'}
kisa = lambda s: s.replace(' Grand Prix', '')
out = {}

# Pilot özeti
out['tahmin'] = pd.DataFrame([dict(
    kod=kodlar[d_ids[j]], pilot=soyad(d_ids[j]), takim=TAKIM[takim_of[d_ids[j]]], mevcut=int(mevcut.get(d_ids[j], 0)),
    beklenen=round(float(toplam[:, j].mean()), 1), p10=round(float(np.percentile(toplam[:, j], 10))),
    p90=round(float(np.percentile(toplam[:, j], 90))), sampiyon=round(float((final_sira[:, j] == 0).mean()), 4),
    ilk3=round(float((final_sira[:, j] < 3).mean()), 4), galibiyet_bek=round(float(wins[:, j].mean()), 2),
    **{f's{k+1}': round(float((final_sira[:, j] == k).mean()), 4) for k in range(8)}) for j in sira_ort])

# Yarış yarış olasılıklar
rows = []
for k, r in kalan.iterrows():
    s, g = yaris_sira[k], yaris_grid[k]
    bo = yaris_ozellik[k].set_index('driver')
    for j in sira_ort:
        d = d_ids[j]
        rows.append(dict(tur=int(r['round']), yaris=kisa(r.race), kod=kodlar[d], pilot=soyad(d),
                         pole=round(float((g[:, j] == 0).mean()), 4), grid_bek=round(float(g[:, j].mean() + 1), 1),
                         kazanma=round(float((s[:, j] == 0).mean()), 4), podyum=round(float((s[:, j] < 3).mean()), 4),
                         puan_alma=round(float((s[:, j] < 10).mean()), 4), dnf=round(float((s[:, j] == 99).mean()), 4),
                         beklenen_puan=round(float(np.where(s[:, j] < n, puan_tablo[np.minimum(s[:, j], n - 1)], 0).mean()), 2),
                         pist_etkisi=round(float(-bo.loc[d, 'benzer_pilot']), 2) if pd.notna(bo.loc[d, 'benzer_pilot']) else 0.0))
out['yarislar'] = pd.DataFrame(rows)

# Pist profilleri (kalan yarışlar)
out['pistler'] = pd.DataFrame([dict(tur=int(r['round']), yaris=kisa(r.race), pist=r.circuit,
                                    **dict(zip(['sokak', 'hiz', 'yuk', 'yukseklik', 'sollama', 'asinma'], PIST[r.circuit])))
                               for _, r in kalan.iterrows()])

# Sezon ilerlemesi
tur_puan = s26.pivot_table(index='round', columns='driver', values='points', aggfunc='sum').fillna(0).cumsum()
rows = []
for d in ilk:
    j = d_ids.index(d)
    rows += [dict(tur=int(t), kod=kodlar[d], pilot=soyad(d), puan=float(v), tip='gercek') for t, v in tur_puan[d].items()]
    rows += [dict(tur=int(kalan['round'][k]), kod=kodlar[d], pilot=soyad(d),
                  puan=round(float(ara_toplam[k][:, j].mean()), 1), tip='tahmin') for k in range(len(kalan))]
out['ilerleme'] = pd.DataFrame(rows)

# Takımlar
takimlar = sorted(set(takim_of.values()))
t_mevcut = s26.groupby('team').points.sum()
T = np.stack([toplam[:, [j for j, d in enumerate(d_ids) if takim_of[d] == t]].sum(axis=1)
              - sum(mevcut.get(d, 0) for d in d_ids if takim_of[d] == t) + t_mevcut.get(t, 0) for t in takimlar], axis=1)
t_sira = (-T).argsort(axis=1).argsort(axis=1)
out['takimlar'] = pd.DataFrame([dict(takim=TAKIM[t], mevcut=int(t_mevcut.get(t, 0)), beklenen=round(float(T[:, i].mean()), 1),
                                     p10=round(float(np.percentile(T[:, i], 10))), p90=round(float(np.percentile(T[:, i], 90))),
                                     sampiyon=round(float((t_sira[:, i] == 0).mean()), 4),
                                     **{f's{k+1}': round(float((t_sira[:, i] == k).mean()), 4) for k in range(5)})
                                for i, t in enumerate(takimlar)]).sort_values('beklenen', ascending=False)

# Puan dağılımı (ilk 6)
rows = []
for d in ilk[:6]:
    v = toplam[:, d_ids.index(d)]
    kenar = np.arange(np.floor(v.min() / 5) * 5, v.max() + 5, 5)
    h, _ = np.histogram(v, bins=kenar)
    rows += [dict(kod=kodlar[d], pilot=soyad(d), dilim=int(a), oran=round(c / N_SIM, 5)) for a, c in zip(kenar[:-1], h)]
out['dagilim'] = pd.DataFrame(rows)

# Kesinleşme: k. yarıştan sonra lider (ya da verilen pilot) her rakibe karşı garanti mi?
# Fark > kalan en fazla puan ise garanti. Fark tam eşitse, puan eşitliğinde galibiyet sayısı belirler:
# rakip kalan yarışların hepsini kazansa bile liderin galibiyeti daha fazlaysa yine garanti.
def kesin_mi(k, j=None):
    T, W = ara_toplam[k], ara_wins[k]
    kalan_yaris = len(kalan) - k - 1
    lid = T.argmax(axis=1) if j is None else np.full(N_SIM, j)
    r_ = np.arange(N_SIM)
    fark = T[r_, lid][:, None] - T
    w_ust = W[r_, lid][:, None] > W + kalan_yaris
    ok = (fark > 25 * kalan_yaris) | ((fark == 25 * kalan_yaris) & w_ust)
    ok[r_, lid] = True
    return ok.all(axis=1)


rows, onceki = [], np.zeros(N_SIM, bool)
for k, r in kalan.iterrows():
    kesin = kesin_mi(k)
    rows.append(dict(tur=int(r['round']), yaris=kisa(r.race), bu_yarista=round(float((kesin & ~onceki).mean()), 4),
                     toplam=round(float(kesin.mean()), 4)))
    onceki = kesin
out['kesinlesme'] = pd.DataFrame(rows)

# Meksika: Antonelli'nin bitirdiği sıraya göre şampiyonluğu bugün garantileme şansı
ant = d_ids.index('antonelli')
s0 = yaris_sira[0]
kesin0 = kesin_mi(0, ant)
rows = []
for p in range(1, 12):
    m = (s0[:, ant] == p - 1) if p <= 10 else (s0[:, ant] >= 10)
    rows.append(dict(sira=str(p) if p <= 10 else 'Puan dışı / DNF', olasilik=round(float(m.mean()), 4),
                     kesinlesir=round(float(kesin0[m].mean()), 4) if m.any() else 0.0))
out['meksika'] = pd.DataFrame(rows)

# Doğrulama ve özellik önemi
out['dogrulama'] = pd.DataFrame([dict(yontem=r['yontem'], mae=round(r['mae'], 2), spearman=round(r['spearman'], 3)) for r in dogrulama])
AC = {'grid': 'Başlangıç sırası', 'takim_puan5': 'Takımın son 5 yarış puanı', 'puan5': 'Pilotun son 5 yarış puanı',
      'takim_ort': 'Takımın sezon ortalama sırası', 'sezon_ort': 'Pilotun sezon ortalama sırası', 'dnf_orani': 'Yarış dışı kalma oranı',
      'form6': 'Son 6 yarış ortalama sırası', 'form3': 'Son 3 yarış ortalama sırası', 'yaris_no': 'Sezonun kaçıncı yarışı',
      'benzer_pilot': 'Pilotun benzer pistlerdeki performansı', 'benzer_takim': 'Takımın benzer pistlerdeki performansı',
      'benzer_grid': 'Benzer pistlerde sıralama turu', 'grid5': 'Son 5 yarış ortalama grid', 'grid_ort': 'Sezon ortalama grid',
      'takim_grid5': 'Takımın son 5 yarış gridi', 'p_sokak': 'Pist: sokak pisti', 'p_hiz': 'Pist: hız',
      'p_yuk': 'Pist: aerodinamik yük', 'p_yukseklik': 'Pist: rakım', 'p_sollama': 'Pist: sollama kolaylığı',
      'p_asinma': 'Pist: lastik aşınması'}
out['onem'] = pd.DataFrame([dict(asama=a, ozellik=AC[k], onem=round(float(v), 3))
                            for a, o in (('Yarış', onem_r), ('Sıralama turu', onem_q))
                            for k, v in sorted(o.items(), key=lambda x: -x[1])])
out['kalan'] = kalan[['round', 'race', 'date']].rename(columns={'round': 'tur', 'race': 'yaris', 'date': 'tarih'})

for ad, df in out.items():
    df.to_csv(f'data/m2_{ad}.csv', index=False)
print(out['tahmin'][['pilot', 'mevcut', 'beklenen', 'sampiyon', 'ilk3', 's2']].head(8).to_string(index=False))
print(out['yarislar'].pivot(index='pilot', columns='tur', values='kazanma').loc[[soyad(d) for d in ilk[:6]]])
