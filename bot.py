import time
import random

kisilik = ["agresif","sakin","neşeli"]
class bot():
    def __init__(self, isim):
        self.isim = isim
        self.kisilik = random.choice(kisilik)
    def isle(self, soru):
        i = soru.lower()
        cevap = "BOT: "
        if "merh" in i: cevap += "merhaba! "
        if "zen" in i and (self.kisilik == "agresif" or "sakin"): cevap += "ırkçı!!! "
        if "zen" in i and self.kisilik == "neşeli": cevap += "HAHAHAHA "
        if "saat" in i and (self.kisilik =="sakin" or "neşeli"): cevap += f"işte saat: {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}"
        if "saat" in i and self.kisilik == "agresif": cevap += "söylemiyom yarram "
        if cevap == "BOT: ": cevap += "anlamadım"
        return cevap