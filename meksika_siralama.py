"""
Meksika GP 2026 sıralama turu (qualifying) tahmini — yarış modelinden bağımsız.

Sürüm 2: sıra yerine TUR ZAMANI üzerinden.
  1) Hedef: pilotun en iyi turunun pole zamanına göre farkı (%). Sıra ortalamasından çok daha ince bir hız
     ölçüsü: %0,1 geride kalmak ile %0,5 geride kalmak, ikisi de "3. sıra" olabilir.
  2) Özellikler: son seanslardaki zaman farkı (medyan + ağırlıklı ortalama; tek kötü seansa dayanıklı),
     takımın farkı, takım arkadaşı, ve Meksika etkisi (pilot/takımın geçmiş Meksika'larda, o sezonki
     normal farkına göre ne kadar hızlı/yavaş olduğu).
  3) Simülasyon: tahmine, her pilotun KENDİ son seanslarındaki sapmalarından rastgele çekilen bir sapma
     eklenir (tutarlı pilot az, dengesiz pilot çok oynar; ara sıra gelen kötü seanslar da dahil).
     Sapma ölçeği, test seanslarında pole olasılıklarını en iyi tutturan değere kalibre edilir.
  4) Test: 2025–2026'nın tüm sıralama turları + 2022–2025 Meksika, her seans o tarihten önceki verilerle.
     Eski sürüm (sıra tahmini, herkese aynı belirsizlik) ile karşılaştırılır.

Kullanım:  ./venv/bin/python meksika_siralama.py
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

PIST = "rodriguez"
N_SIM = 20000
N_TEST_SIM = 3000
rng = np.random.default_rng(7)

q = pd.read_csv("data/siralama.csv")
q["idx"] = q.season * 100 + q["round"]
q["en_iyi"] = q[["q1", "q2", "q3"]].min(axis=1)
q["fark"] = (q.en_iyi / q.groupby("idx").en_iyi.transform("min") - 1) * 100
q["fark"] = q.fark.fillna(4.0).clip(upper=4.0)        # tur atamayan / aşırı yavaş: üst sınır
q["sez_med"] = q.groupby(["season", "driver"]).fark.transform("median")
q["t_sez_med"] = q.groupby(["season", "team"]).fark.transform("median")
q["p_sapma"] = q.pos - q.groupby(["season", "driver"]).pos.transform("mean")
q["t_sapma"] = q.pos - q.groupby(["season", "team"]).pos.transform("mean")
order = sorted(q.idx.unique())
seans_sirasi = {ix: i for i, ix in enumerate(order)}
q["sn"] = q.idx.map(seans_sirasi)


def ozellikler(gecmis, pilotlar, sezon, pist, simdi):
    """Yalnızca `gecmis` (hedef seanstan önceki seanslar) kullanılır."""
    bu = gecmis[gecmis.season == sezon]
    son = bu if bu.idx.nunique() >= 3 else gecmis[gecmis.season >= sezon - 1]
    son_ix = sorted(son.idx.unique())
    son8 = son[son.idx.isin(son_ix[-8:])]
    t_med8 = son8.groupby("team").fark.median()
    t_pos = son.groupby("team").pos.mean()
    pg = gecmis[(gecmis.circuit == pist) & (gecmis.season < sezon)]
    pg = pg.assign(w=0.75 ** (sezon - 1 - pg.season),
                   z_sapma=pg.fark - pg.sez_med, tz_sapma=pg.fark - pg.t_sez_med)
    rows = []
    for d, t in pilotlar:
        g = son[son.driver == d].sort_values("idx")
        g8 = g.tail(8)
        w = 0.8 ** (simdi - g8.sn) if len(g8) else None
        # tek kötü seans (kaza, tur atamama) ortalamayı bozmasın: medyanın 0,6 puan üstü kırpılır
        g8c = g8.fark.clip(upper=g8.fark.median() + 0.6) if len(g8) else g8.fark
        pd_, pt_ = pg[pg.driver == d], pg[pg.team == t]
        rows.append(dict(
            driver=d, team=t,
            # zaman tabanlı form
            fark_med8=g8.fark.median(), fark_ew=(g8c * w).sum() / w.sum() if len(g8) else np.nan,
            fark_son3=g.fark.tail(3).median(), fark_min5=g.fark.tail(5).min(),
            t_fark_med8=t_med8.get(t, np.nan),
            takim_ici=(g8.fark.median() - t_med8.get(t, np.nan)) if len(g8) else np.nan,
            # sıra tabanlı form (eski sürümdeki gibi)
            q_form5=g.pos.tail(5).mean(), q_sezon=g.pos.mean(), t_sezon=t_pos.get(t, np.nan),
            seans_no=bu.idx.nunique(),
            # Meksika (pist) etkisi
            pist_zaman_p=(pd_.z_sapma * pd_.w).sum() / pd_.w.sum() if len(pd_) else np.nan,
            pist_zaman_t=(pt_.tz_sapma * pt_.w).sum() / pt_.w.sum() if len(pt_) else np.nan,
            pist_sira_p=(pd_.p_sapma * pd_.w).sum() / pd_.w.sum() if len(pd_) else np.nan,
            pist_sira_t=(pt_.t_sapma * pt_.w).sum() / pt_.w.sum() if len(pt_) else np.nan,
            pist_deneyim=pd_.season.nunique(),
        ))
    return pd.DataFrame(rows)


ZAMAN = ["fark_med8", "fark_ew", "fark_son3", "fark_min5", "t_fark_med8", "takim_ici", "seans_no",
         "pist_zaman_p", "pist_zaman_t", "pist_deneyim"]
ESKI = ["q_form5", "q_sezon", "t_sezon", "seans_no", "pist_sira_p", "pist_sira_t", "pist_deneyim"]

# ---- tüm seanslar için özellikler (2017'den itibaren) ----
X = []
for ix in order:
    if ix < 201701:
        continue
    h = q[q.idx == ix]
    f = ozellikler(q[q.idx < ix], list(zip(h.driver, h.team)), h.season.iloc[0], h.circuit.iloc[0], seans_sirasi[ix])
    f["pos"], f["fark"], f["idx"] = h.pos.values, h.fark.values, ix
    f["circuit"], f["season"], f["sn"] = h.circuit.iloc[0], h.season.iloc[0], seans_sirasi[ix]
    X.append(f)
data = pd.concat(X, ignore_index=True)
MED = {c: data[c].median() for c in set(ZAMAN + ESKI)}
for c in ["pist_zaman_p", "pist_zaman_t", "pist_sira_p", "pist_sira_t"]:
    MED[c] = 0.0  # pist geçmişi yoksa etki yok say
data = data.fillna(MED)


import os
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Karisik:
    """Zaman modeli: doğrusal (Huber, uç değerlere dayanıklı) — en hızlı pilotları ortalamaya çekmez."""
    def __init__(self):
        self.m = make_pipeline(StandardScaler(), HuberRegressor(alpha=1.0, max_iter=500))
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        self.m.fit(X, y, huberregressor__sample_weight=sample_weight)
        c = np.abs(self.m[-1].coef_)
        self.feature_importances_ = c / c.sum()
        return self

    def predict(self, X):
        return self.m.predict(X)


MODEL_TURU = os.environ.get("MODEL", "huber")


def rf(zaman=False):
    if zaman and MODEL_TURU == "huber":
        return Karisik()
    return RandomForestRegressor(n_estimators=250, min_samples_leaf=10, random_state=0, n_jobs=-1)


def agirlik(df):
    return 0.8 ** (2026 - df.season)


def sapmalar(gecmis, d, simdi):
    """Pilotun son 12 seansında kendi medyanından sapmaları (zaman, %)."""
    g = gecmis[gecmis.driver == d].sort_values("idx").tail(12)
    return (g.fark - g.fark.median()).values if len(g) >= 5 else None


def simule(bek, gecmis, pilotlar, simdi, olcek, n_sim, havuz):
    n = len(pilotlar)
    S = np.empty((n_sim, n))
    for j, (d, _) in enumerate(pilotlar):
        s = sapmalar(gecmis, d, simdi)
        s = havuz if s is None else s
        # sapma: %50 pilotun kendi geçmişinden, %50 tüm pilotların havuzundan (tek şanslı seansa aşırı güvenmesin)
        kendi = rng.random(n_sim) < 0.5
        S[:, j] = bek[j] + olcek * np.where(kendi, rng.choice(s, n_sim), rng.choice(havuz, n_sim))
    return S.argsort(axis=1).argsort(axis=1) + 1  # 1 = pole


# ---- test: 2025–2026 tüm seanslar + 2022–2024 Meksika ----
test_ix = [ix for ix in order if ix >= 202501 and ix < 202619]
test_ix += [ix for ix in data[(data.circuit == PIST) & data.season.between(2022, 2024)].idx.unique()]
test_ix = sorted(set(test_ix))
OLCEKLER = [0.7, 1.0, 1.3, 1.6]
kayit = []  # her test seansı için tahminler
for ix in test_ix:
    tr, te = data[data.sn < seans_sirasi[ix]], data[data.idx == ix]
    gecmis = q[q.idx < ix]
    pil = list(zip(te.driver, te.team))
    mz = rf(True).fit(tr[ZAMAN], tr.fark, sample_weight=agirlik(tr))
    me = rf().fit(tr[ESKI], tr.pos, sample_weight=agirlik(tr))
    bz, be = mz.predict(te[ZAMAN]), me.predict(te[ESKI])
    havuz = (tr.fark - tr.fark_med8).clip(-1.5, 2.5).values[-3000:]
    pole_gercek = (te.pos.values == 1)
    r = dict(idx=ix, mexico=te.circuit.iloc[0] == PIST, gercek=te.pos.values,
             zaman_bek=bz, eski_bek=be, form=te.q_form5.values)
    for o in OLCEKLER:
        r[f"pole_z_{o}"] = (simule(bz, gecmis, pil, seans_sirasi[ix], o, N_TEST_SIM, havuz) == 1).mean(axis=0)
    # eski sürüm: sıra tahmini + herkese aynı normal belirsizlik (±3,95 sıra)
    es = be + rng.normal(0, 3.95, (N_TEST_SIM, len(pil)))
    r["pole_e"] = ((es.argsort(axis=1).argsort(axis=1) + 1) == 1).mean(axis=0)
    r["pole_gercek"] = pole_gercek
    kayit.append(r)


def ozet(kayitlar, en_iyi_olcek):
    out = []
    for ad, bek_k, pole_k in (("Son 5 sıralama ortalaması", "form", None),
                              ("Eski model (sıra tahmini)", "eski_bek", "pole_e"),
                              ("Yeni model (tur zamanı)", "zaman_bek", f"pole_z_{en_iyi_olcek}")):
        sp = np.mean([spearmanr(k[bek_k], k["gercek"]).statistic for k in kayitlar])
        mae = np.mean([np.abs(pd.Series(k[bek_k]).rank().values - k["gercek"]).mean() for k in kayitlar])
        pole_bildi = sum(int(np.argmin(k[bek_k]) == np.argmax(k["pole_gercek"])) for k in kayitlar)
        brier = np.mean([((k[pole_k] - k["pole_gercek"]) ** 2).sum() for k in kayitlar]) if pole_k else np.nan
        pole_ihtimal = np.mean([k[pole_k][k["pole_gercek"]].sum() for k in kayitlar]) if pole_k else np.nan
        ilk3 = np.mean([len(set(np.argsort(k[bek_k])[:3]) & set(np.where(k["gercek"] <= 3)[0])) for k in kayitlar])
        out.append(dict(yontem=ad, seans=len(kayitlar), sira_hatasi=round(mae, 2), uyum=round(sp, 3),
                        pole_bildi=pole_bildi, ilk3_isabet=round(ilk3, 2),
                        pole_brier=round(brier, 3) if pole_k else None,
                        poleciye_verilen=round(pole_ihtimal, 3) if pole_k else None))
    return pd.DataFrame(out)


# ölçek kalibrasyonu: pole olasılıklarının Brier skoru en düşük olan
brier_o = {o: np.mean([((k[f"pole_z_{o}"] - k["pole_gercek"]) ** 2).sum() for k in kayit]) for o in OLCEKLER}
OLCEK = min(brier_o, key=brier_o.get)
pd.set_option("display.width", 220)
print("Sapma ölçeği kalibrasyonu (pole Brier, düşük iyi):", {k: round(v, 4) for k, v in brier_o.items()}, "->", OLCEK)
dog_tum = ozet(kayit, OLCEK).assign(test="2025–2026 tüm seanslar + 2022–24 Meksika")
dog_mex = ozet([k for k in kayit if k["mexico"]], OLCEK).assign(test="Sadece Meksika 2022–2024")
dogrulama = pd.concat([dog_tum, dog_mex], ignore_index=True)
print("\nTest (her seans, o tarihten önceki verilerle):")
print(dogrulama.to_string(index=False))

# Meksika etkisi işe yarıyor mu? (aynı test, pist özellikleri olmadan)
ZAMAN_PISTSIZ = [c for c in ZAMAN if not c.startswith("pist")]
sp_p, sp_s = [], []
for k in kayit:
    ix = k["idx"]
    tr, te = data[data.sn < seans_sirasi[ix]], data[data.idx == ix]
    sp_s.append(spearmanr(rf(True).fit(tr[ZAMAN_PISTSIZ], tr.fark, sample_weight=agirlik(tr)).predict(te[ZAMAN_PISTSIZ]), te.pos).statistic)
    sp_p.append(spearmanr(k["zaman_bek"], k["gercek"]).statistic)
mx = [i for i, k in enumerate(kayit) if k["mexico"]]
pist_katkisi = dict(tum_pistli=np.mean(sp_p), tum_pistsiz=np.mean(sp_s),
                    mex_pistli=np.mean([sp_p[i] for i in mx]), mex_pistsiz=np.mean([sp_s[i] for i in mx]))
print("\nPist geçmişinin katkısı (uyum):", {k: round(v, 3) for k, v in pist_katkisi.items()})

# ---- 2026 Meksika tahmini ----
m = rf(True).fit(data[ZAMAN], data.fark, sample_weight=agirlik(data))
son = q[q.idx == q[q.season == 2026].idx.max()]
pilotlar = list(zip(son.driver, son.team))
simdi = len(order)
f = ozellikler(q, pilotlar, 2026, PIST, simdi).fillna(MED)
bek = m.predict(f[ZAMAN])
havuz = (data.fark - data.fark_med8).clip(-1.5, 2.5).values[-3000:]
sira = simule(bek, q, pilotlar, simdi, OLCEK, N_SIM, havuz)
n = len(pilotlar)
q1_kesim = n - 6  # 22 araçta Q1'de son 6 elenir, Q2'de sonraki 6
kod = dict(zip(q.driver, q.code))
TAKIM = {'mercedes': 'Mercedes', 'ferrari': 'Ferrari', 'red_bull': 'Red Bull', 'mclaren': 'McLaren', 'rb': 'Racing Bulls',
         'alpine': 'Alpine', 'williams': 'Williams', 'haas': 'Haas', 'audi': 'Audi', 'cadillac': 'Cadillac',
         'aston_martin': 'Aston Martin'}
q26 = q[q.season == 2026]
out = pd.DataFrame(dict(
    kod=[kod[d] for d, _ in pilotlar], pilot=[d.split('_')[-1].capitalize() for d, _ in pilotlar],
    takim=[TAKIM.get(t, t) for _, t in pilotlar],
    tahmini_fark_pct=bek.round(3), beklenen_sira=sira.mean(axis=0).round(1),
    pole=(sira == 1).mean(axis=0).round(4), on_sira=(sira <= 2).mean(axis=0).round(4),
    ilk3=(sira <= 3).mean(axis=0).round(4), q3=(sira <= 10).mean(axis=0).round(4),
    q1_elenme=(sira > q1_kesim).mean(axis=0).round(4),
    pole_2026=[int(((q26.driver == d) & (q26.pos == 1)).sum()) for d, _ in pilotlar],
    fark_med8=f.fark_med8.round(3).values,
    meksika_zaman_etkisi=f.pist_zaman_p.round(3).values,   # eksi: Meksika'da normalinden hızlı
    takim_meksika_zaman_etkisi=f.pist_zaman_t.round(3).values,
    meksika_deneyim=f.pist_deneyim.astype(int).values,
)).sort_values("beklenen_sira").reset_index(drop=True)

gec = q[(q.circuit == PIST) & q.driver.isin([d for d, _ in pilotlar])]
gecmis_tablo = gec.pivot_table(index="driver", columns="season", values="pos").reindex([d for d, _ in pilotlar])
gecmis_tablo.index = [kod[d] for d in gecmis_tablo.index]
onem = pd.Series(m.feature_importances_, index=ZAMAN).sort_values(ascending=False)

print(f"\n2026 Meksika sıralama turu tahmini (sapma ölçeği {OLCEK}):")
print(out.to_string(index=False))
print("\nÖzellik önemi:\n", onem.round(3).to_string())

out.to_csv("data/meksika_siralama_tahmin.csv", index=False)
gecmis_tablo.reset_index(names="kod").to_csv("data/meksika_siralama_gecmis.csv", index=False)
dogrulama.to_csv("data/meksika_siralama_dogrulama.csv", index=False)
onem.round(4).rename("onem").rename_axis("ozellik").reset_index().to_csv("data/meksika_siralama_onem.csv", index=False)
