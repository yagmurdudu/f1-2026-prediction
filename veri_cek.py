"""2025 ve 2026 F1 yarış sonuçlarını Jolpica (Ergast) API'den indirir -> data/*.csv"""
import time, os, requests, pandas as pd

API = "https://api.jolpi.ca/ergast/f1"
os.makedirs("data", exist_ok=True)


def get(url):
    for _ in range(5):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:  # hız sınırı
            time.sleep(3)
            continue
        r.raise_for_status()
        return r.json()["MRData"]
    raise RuntimeError(url)


def takvim(yil):
    races = get(f"{API}/{yil}.json?limit=100")["RaceTable"]["Races"]
    return [dict(season=yil, round=int(r["round"]), race=r["raceName"],
                 circuit=r["Circuit"]["circuitId"], date=r["date"],
                 sprint="Sprint" in r) for r in races]


def sonuclar(yil, rnd, tur="results"):
    d = get(f"{API}/{yil}/{rnd}/{tur}.json?limit=100")["RaceTable"]["Races"]
    if not d:
        return []
    key = "Results" if tur == "results" else "SprintResults"
    out = []
    for x in d[0][key]:
        out.append(dict(season=yil, round=rnd, kind="race" if tur == "results" else "sprint",
                        driver=x["Driver"]["driverId"], code=x["Driver"].get("code", ""),
                        team=x["Constructor"]["constructorId"], grid=int(x["grid"]),
                        pos=int(x["position"]), points=float(x["points"]),
                        status=x["status"], classified=x["positionText"].isdigit()))
    return out


cal, rows = [], []
for yil in (2025, 2026):
    t = takvim(yil)
    cal += t
    for r in t:
        if r["date"] > time.strftime("%Y-%m-%d"):
            continue
        rows += sonuclar(yil, r["round"])
        if r["sprint"]:
            rows += sonuclar(yil, r["round"], "sprint")
        time.sleep(0.3)
    print(yil, "tamam")

pd.DataFrame(cal).to_csv("data/takvim.csv", index=False)
df = pd.DataFrame(rows)
df.to_csv("data/sonuclar.csv", index=False)
print(df.groupby(["season", "kind"])["round"].nunique())
