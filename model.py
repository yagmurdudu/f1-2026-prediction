"""
F1 2026 şampiyonluk tahmini.
1) Her pilot + yarış için yalnızca o yarıştan ÖNCEKİ verilerden özellikler üretilir (form, takım gücü, pist geçmişi...).
2) Gradient Boosting modeli bitiriş sırasını tahmin etmeyi öğrenir.
3) Kalan yarışlar Monte Carlo ile 20.000 kez simüle edilir -> şampiyonluk / sıralama olasılıkları.
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

PUAN = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
N_SIM = 20000
rng = np.random.default_rng(42)

res = pd.read_csv("data/sonuclar.csv")
cal = pd.read_csv("data/takvim.csv")
races = res[res.kind == "race"].merge(cal[["season", "round", "circuit"]], on=["season", "round"])
races["dnf"] = (~races.classified).astype(int)
races["idx"] = races.season * 100 + races["round"]  # kronolojik sıra
order = sorted(races.idx.unique())


def ozellikler(gecmis, pilotlar, sezon, circuit):
    """gecmis: hedef yarıştan önceki tüm yarışlar. pilotlar: [(driver, team)]"""
    bu_sezon = gecmis[gecmis.season == sezon]
    onceki = gecmis[gecmis.season == sezon - 1]
    son = bu_sezon if len(bu_sezon) else onceki
    takim_son = son[son.idx.isin(sorted(son.idx.unique())[-5:])].groupby("team").points.mean()
    takim_sezon = son.groupby("team").pos.mean()
    rows = []
    for d, t in pilotlar:
        g = son[son.driver == d].sort_values("idx")
        p = onceki[(onceki.driver == d) & (onceki.circuit == circuit)]
        rows.append(dict(
            driver=d, team=t,
            form3=g.pos.tail(3).mean(), form6=g.pos.tail(6).mean(),
            puan5=g.points.tail(5).mean(), sezon_ort=g.pos.mean(),
            takim_puan5=takim_son.get(t, np.nan), takim_ort=takim_sezon.get(t, np.nan),
            dnf_orani=(g.dnf.sum() + 1) / (len(g) + 10),   # yumuşatılmış
            pist_gecmis=p.pos.mean() if len(p) else np.nan,
            yaris_no=len(bu_sezon.idx.unique()),
        ))
    return pd.DataFrame(rows)


# ---- eğitim seti ----
X_list = []
for i, ix in enumerate(order[1:], 1):
    hedef = races[races.idx == ix]
    gecmis = races[races.idx < ix]
    s, c = hedef.season.iloc[0], hedef.circuit.iloc[0]
    f = ozellikler(gecmis, list(zip(hedef.driver, hedef.team)), s, c)
    f["pos"] = hedef.pos.values
    f["idx"] = ix
    X_list.append(f)
data = pd.concat(X_list, ignore_index=True)
FEATS = ["form3", "form6", "puan5", "sezon_ort", "takim_puan5", "takim_ort", "dnf_orani", "pist_gecmis", "yaris_no"]
for col in FEATS:  # yeni pilot / yeni takım boşlukları
    data[col] = data[col].fillna(data[col].median())


def model_kur():
    # Doğrulamada GBR / Ridge / RF denendi; sıralamayı en iyi RF tahmin etti.
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=0)


# ---- doğrulama: 2026'nın son 6 yarışını hiç görmeden tahmin et ----
test_ix = [ix for ix in order if ix >= 202613]
tr, te = data[~data.idx.isin(test_ix)], data[data.idx.isin(test_ix)]
w = np.where(tr.idx > 202600, 3.0, 1.0)  # 2026 (yeni kurallar) verisine daha çok ağırlık
m = model_kur().fit(tr[FEATS], tr.pos, sample_weight=w)
pred = m.predict(te[FEATS])
baz = te.form6.values  # basit tahmin: son 6 yarış ortalaması
sp_m = np.mean([spearmanr(pred[te.idx == ix], te.pos[te.idx == ix]).statistic for ix in test_ix])
sp_b = np.mean([spearmanr(baz[te.idx == ix], te.pos[te.idx == ix]).statistic for ix in test_ix])
dogrulama = dict(model_mae=mean_absolute_error(te.pos, pred), baz_mae=mean_absolute_error(te.pos, baz),
                 model_spearman=sp_m, baz_spearman=sp_b)
print("Doğrulama (2026 R13-R18):", {k: round(v, 3) for k, v in dogrulama.items()})

# ---- final model: tüm veri ----
w = np.where(data.idx > 202600, 3.0, 1.0)
model = model_kur().fit(data[FEATS], data.pos, sample_weight=w)
# belirsizlik: modelin görmediği yarışlardaki (doğrulama) hata, DNF'ler hariç (onlar ayrıca simüle ediliyor)
bitiren = (te.pos <= 18).values
resid_std = float(np.std(te.pos.values[bitiren] - pred[bitiren]))
onem = dict(sorted(zip(FEATS, model.feature_importances_.round(3)), key=lambda x: -x[1]))
print("Özellik önemi:", onem, "| artık std:", round(resid_std, 2))

# ---- kalan yarışlar ----
son_yaris = races[races.idx == order[-1]]
pilotlar = list(zip(son_yaris.driver, son_yaris.team))
kodlar = dict(zip(res.driver, res.code))
kalan = cal[(cal.season == 2026) & (cal["round"] > races[races.season == 2026]["round"].max())]
mevcut = res[res.season == 2026].groupby("driver").points.sum()
galibiyet = res[(res.season == 2026) & (res.kind == "race") & (res.pos == 1)].driver.value_counts()

d_ids = [d for d, _ in pilotlar]
n = len(d_ids)
beklenen = []  # her kalan yarış için tahmini sıra (gürültüsüz)
dnf_p = None
for _, r in kalan.iterrows():
    f = ozellikler(races, pilotlar, 2026, r.circuit)
    for col in FEATS:
        f[col] = f[col].fillna(data[col].median())
    beklenen.append(model.predict(f[FEATS]))
    dnf_p = f.dnf_orani.values
beklenen = np.array(beklenen)  # (yarış, pilot)

toplam = np.tile([mevcut.get(d, 0) for d in d_ids], (N_SIM, 1)).astype(float)
wins = np.tile([galibiyet.get(d, 0) for d in d_ids], (N_SIM, 1)).astype(float)
puan_tablo = np.array([PUAN.get(k, 0) for k in range(1, n + 1)])
yaris_sira, ara_toplam = [], []  # her yarışın sırası ve yarış sonrası toplamlar
for k in range(len(kalan)):
    skor = beklenen[k] + rng.normal(0, resid_std, (N_SIM, n))
    dnf = rng.random((N_SIM, n)) < dnf_p
    skor[dnf] += 100  # yarış dışı kalma
    sira = skor.argsort(axis=1).argsort(axis=1)  # 0 = birinci
    sira[dnf] = 99  # yarış dışı kalan puan almaz
    toplam += np.where(sira < n, puan_tablo[np.minimum(sira, n - 1)], 0)
    wins += (sira == 0)
    yaris_sira.append(sira.copy())
    ara_toplam.append(toplam.copy())

# eşitlikte galibiyet sayısı belirler
final_sira = (-(toplam + wins * 1e-3)).argsort(axis=1).argsort(axis=1)
sonuc = []
for j, d in enumerate(d_ids):
    sonuc.append(dict(
        driver=d, code=kodlar.get(d, d[:3].upper()), team=dict(pilotlar)[d],
        mevcut=float(mevcut.get(d, 0)), beklenen_final=float(toplam[:, j].mean()),
        p10=float(np.percentile(toplam[:, j], 10)), p90=float(np.percentile(toplam[:, j], 90)),
        sampiyon=float((final_sira[:, j] == 0).mean()),
        ilk3=float((final_sira[:, j] < 3).mean()),
        sira_dagilim=[float((final_sira[:, j] == k).mean()) for k in range(8)],
        yaris_tahmin=[float(x) for x in beklenen[:, j]],
    ))
sonuc.sort(key=lambda x: -x["beklenen_final"])

print(f"\n{'Pilot':<16}{'Şimdi':>6}{'Beklenen':>10}{'Şampiyon %':>12}{'İlk 3 %':>9}")
for s in sonuc[:10]:
    print(f"{s['driver']:<16}{s['mevcut']:>6.0f}{s['beklenen_final']:>10.1f}{s['sampiyon']*100:>12.2f}{s['ilk3']*100:>9.1f}")

json.dump(dict(sonuc=sonuc, kalan=kalan[["round", "race", "date"]].to_dict("records"),
               dogrulama=dogrulama, onem={k: float(v) for k, v in onem.items()},
               son_round=int(races[races.season == 2026]["round"].max()), n_sim=N_SIM),
          open("data/tahmin.json", "w"), ensure_ascii=False, indent=1)

# ================= Detaylı çıktılar (sayfa için CSV) =================
TAKIM = {'mercedes': 'Mercedes', 'ferrari': 'Ferrari', 'red_bull': 'Red Bull', 'mclaren': 'McLaren',
         'rb': 'Racing Bulls', 'alpine': 'Alpine', 'williams': 'Williams', 'haas': 'Haas', 'audi': 'Audi',
         'cadillac': 'Cadillac', 'aston_martin': 'Aston Martin'}
soyad = lambda d: d.split('_')[-1].capitalize()
ilk = [s['driver'] for s in sonuc[:8]]          # öne çıkan 8 pilot
ilk_j = [d_ids.index(d) for d in ilk]

# 1) Sezon boyunca puan (gerçek) + kalan yarışlar için beklenen (tahmin)
s26 = res[res.season == 2026].copy()
tur_puan = s26.pivot_table(index='round', columns='driver', values='points', aggfunc='sum').fillna(0).cumsum()
satirlar = []
for d in ilk:
    for rnd, v in tur_puan[d].items():
        satirlar.append(dict(tur=int(rnd), kod=kodlar[d], pilot=soyad(d), puan=float(v), tur_tipi='gercek'))
    j = d_ids.index(d)
    for k, (_, r) in enumerate(kalan.iterrows()):
        satirlar.append(dict(tur=int(r['round']), kod=kodlar[d], pilot=soyad(d),
                             puan=round(float(ara_toplam[k][:, j].mean()), 1), tur_tipi='tahmin'))
pd.DataFrame(satirlar).to_csv('data/ilerleme.csv', index=False)

# 2) Kalan her yarış için olasılıklar
satirlar = []
for k, (_, r) in enumerate(kalan.iterrows()):
    s = yaris_sira[k]
    for j in ilk_j + [jj for jj in range(n) if jj not in ilk_j]:
        satirlar.append(dict(tur=int(r['round']), yaris=r['race'].replace(' Grand Prix', ''), kod=kodlar[d_ids[j]],
                             pilot=soyad(d_ids[j]), kazanma=round(float((s[:, j] == 0).mean()), 4),
                             podyum=round(float((s[:, j] < 3).mean()), 4),
                             puan_alma=round(float((s[:, j] < 10).mean()), 4),
                             dnf=round(float((s[:, j] == 99).mean()), 4),
                             beklenen_puan=round(float(np.where(s[:, j] < n, puan_tablo[np.minimum(s[:, j], n - 1)], 0).mean()), 2)))
pd.DataFrame(satirlar).to_csv('data/yarislar.csv', index=False)

# 3) Takımlar şampiyonası
takim_of = dict(pilotlar)
takimlar = sorted(set(takim_of.values()))
t_mevcut = s26.groupby('team').points.sum()
T = np.stack([toplam[:, [j for j, d in enumerate(d_ids) if takim_of[d] == t]].sum(axis=1)
              - sum(mevcut.get(d, 0) for d in d_ids if takim_of[d] == t) + t_mevcut.get(t, 0) for t in takimlar], axis=1)
t_sira = (-T).argsort(axis=1).argsort(axis=1)
satirlar = []
for i, t in enumerate(takimlar):
    satirlar.append(dict(takim=TAKIM.get(t, t), mevcut=int(t_mevcut.get(t, 0)), beklenen=round(float(T[:, i].mean()), 1),
                         p10=round(float(np.percentile(T[:, i], 10))), p90=round(float(np.percentile(T[:, i], 90))),
                         sampiyon=round(float((t_sira[:, i] == 0).mean()), 4),
                         **{f's{k+1}': round(float((t_sira[:, i] == k).mean()), 4) for k in range(5)}))
pd.DataFrame(satirlar).sort_values('beklenen', ascending=False).to_csv('data/takimlar.csv', index=False)

# 4) Final puan dağılımı (ilk 6 pilot), 5 puanlık dilimler
satirlar = []
for d in ilk[:6]:
    j = d_ids.index(d)
    v = toplam[:, j]
    kenar = np.arange(np.floor(v.min() / 5) * 5, v.max() + 5, 5)
    h, _ = np.histogram(v, bins=kenar)
    for a, c in zip(kenar[:-1], h):
        satirlar.append(dict(kod=kodlar[d], pilot=soyad(d), dilim=int(a), oran=round(c / N_SIM, 5)))
pd.DataFrame(satirlar).to_csv('data/dagilim.csv', index=False)

# 5) Şampiyonluk hangi yarışta kesinleşir? (lider farkı > kalan en fazla puan)
satirlar, onceki = [], np.zeros(N_SIM, bool)
for k, (_, r) in enumerate(kalan.iterrows()):
    tt = np.sort(ara_toplam[k], axis=1)
    kalan_max = 25 * (len(kalan) - k - 1)
    kesin = (tt[:, -1] - tt[:, -2]) > kalan_max
    satirlar.append(dict(tur=int(r['round']), yaris=r['race'].replace(' Grand Prix', ''),
                         bu_yarista=round(float((kesin & ~onceki).mean()), 4), toplam=round(float(kesin.mean()), 4)))
    onceki = kesin
pd.DataFrame(satirlar).to_csv('data/kesinlesme.csv', index=False)

# 6) Meksika senaryoları: Antonelli bugün şampiyonluğu garantiler mi?
ant, rus = d_ids.index('antonelli'), d_ids.index('russell')
s0 = yaris_sira[0]
fark0 = ara_toplam[0][:, ant] - np.delete(ara_toplam[0], ant, axis=1).max(axis=1)
satirlar = []
for p in range(1, 11):
    m = s0[:, ant] == p - 1
    satirlar.append(dict(antonelli_sira=p, olasilik=round(float(m.mean()), 4),
                         kesinlesir=round(float((fark0[m] > 25 * (len(kalan) - 1)).mean()) if m.any() else 0, 4)))
m = s0[:, ant] >= 10
satirlar.append(dict(antonelli_sira=11, olasilik=round(float(m.mean()), 4),
                     kesinlesir=round(float((fark0[m] > 25 * (len(kalan) - 1)).mean()) if m.any() else 0, 4)))
pd.DataFrame(satirlar).to_csv('data/meksika.csv', index=False)
print('Detaylı CSV dosyaları yazıldı.')
