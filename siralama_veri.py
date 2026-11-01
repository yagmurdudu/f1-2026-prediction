"""2015–2026 tüm sıralama turu (qualifying) sonuçlarını Jolpica F1 API'den indirir -> data/siralama.csv"""
import time, requests, pandas as pd

API = "https://api.jolpi.ca/ergast/f1"


def get(url):
    for _ in range(6):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:  # hız sınırı
            time.sleep(4)
            continue
        r.raise_for_status()
        return r.json()["MRData"]
    raise RuntimeError(url)


def sure(s):
    """'1:17.123' -> saniye; boşsa None"""
    if not s:
        return None
    try:
        m, sn = s.split(":") if ":" in s else ("0", s)
        return int(m) * 60 + float(sn)
    except ValueError:
        return None


rows = []
for yil in range(2015, 2027):
    offset, toplam = 0, 1
    while offset < toplam:
        d = get(f"{API}/{yil}/qualifying.json?limit=100&offset={offset}")
        toplam = int(d["total"])
        for race in d["RaceTable"]["Races"]:
            for q in race["QualifyingResults"]:
                rows.append(dict(
                    season=yil, round=int(race["round"]), circuit=race["Circuit"]["circuitId"],
                    race=race["raceName"], driver=q["Driver"]["driverId"], code=q["Driver"].get("code", ""),
                    team=q["Constructor"]["constructorId"], pos=int(q["position"]),
                    q1=sure(q.get("Q1")), q2=sure(q.get("Q2")), q3=sure(q.get("Q3"))))
        offset += 100
        time.sleep(0.4)
    print(yil, "tamam")

df = pd.DataFrame(rows).drop_duplicates(["season", "round", "driver"])
df.to_csv("data/siralama.csv", index=False)
print(len(df), "satır;", df.groupby("season")["round"].nunique().to_dict())
