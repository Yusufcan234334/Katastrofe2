from flask import Flask, request
from bot import bot
import os
import json

app = Flask(__name__)
veridosyasi = "veritabani/veri.json"

if os.path.exists(veridosyasi):
    print("tamam dosya var")
    with open(veridosyasi, "r", encoding="utf-8") as dosya:
        try:
            olusturulmusbotlar = json.load(dosya)
        except:
            print("Dosya boş yeni sözlük geliyor adamım!")
            olusturulmusbotlar = {}


else:
    print("dosya yok oluşturuluyor")
    with open(veridosyasi, "x", encoding="utf-8") as dosya:
        print("Dosya var oldu")
        olusturulmusbotlar = {}

@app.route("/oluştur", methods=["POST"])
def olustur():
    gelenisteginverisi = request.json
    isim = gelenisteginverisi["isim"]
    bot1 = bot(isim)
    olusturulmusbotlar[bot1.isim] = bot1.__dict__
    with open(veridosyasi, "w", encoding="utf-8") as dosya:
        json.dump(olusturulmusbotlar, dosya, ensure_ascii=False, indent=4)

    return f"""Oluşturulan BOT: {bot1.isim}
Şuan ki kişiliği: {bot1.kisilik}
    """
@app.route("/sor", methods=["POST"])
def sor():
    veri = request.json
    isim = veri["isim"]
    if isim not in olusturulmusbotlar: return "Bot bulunamadı"
    else:
        botismi = olusturulmusbotlar[isim]
        canlananbot = bot(botismi)
        mesaj = veri["mesaj"]
        return canlananbot.isle(mesaj)


@app.route("/", methods=["POST"])
def test():
    return "çalışıyo lannnnnn"



app.run(host="0.0.0.0", port=5000, debug=True)