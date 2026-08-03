import requests
import time

print("Botname'ye Hoşgeldiniz!")
print("Yükleniyor...")
def olustur():
    isim = input("Bot İsmi ?")
    bot = requests.post(url="http://127.0.0.1:5000/oluştur", json={
        "isim": isim
    })
    print(bot.text)
print("%30")
def sohbetdongu():
    print("Çıkmak için q yaz")
    bisim = input("Bot ismi ?")
    while True:
        mesaj = input("Sen: ")
        if mesaj == "q":
            print("BOT: bye")
            time.sleep(3)
            main()

        cevap = requests.post(url="http://127.0.0.1:5000/sor", json={
            "isim": bisim,
            "mesaj": mesaj
        })
        print(cevap.text)
print("%100")
print("Yüklendi!")
print("==================================================")


def main():

    secim = input("Ne yapmak istersin ? 1: oluştur, 2: sohbet döngüsü ")

    if secim == "1":
        olustur()
        main()
    elif secim == "2":
        sohbetdongu()
        main()
    else:
        print("Geçersiz seçim.")
        main()

if __name__ == "__main__":
    main()
