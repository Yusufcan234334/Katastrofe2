import torch
import torch.nn as siniragi
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
seed = 95
trainseed = 33
torch.manual_seed(seed)
print("Veri yükleniyor...")
egitimverisi = pd.read_csv("titianicdatas/train.csv")
print("%50")
testverisi = pd.read_csv("titianicdatas/test.csv")
print("Veri yüklendi!")
print("Veri temizleniyor....")
debugdenemesifln = True
mod = "normal"
egitimtamammi = False

unvanlar = {
    "Mr": 1,
    "Miss": 2,
    "Mrs": 3,
    "Master": 4,
    "Dr": 5,
    "Rev": 6,
    "Col": 7,
    "Major": 8,
    "Mlle": 9,
    "Countess": 10,
    "Ms": 11,
    "Lady": 12,
    "Jonkheer": 13,
    "Don": 14,
    "Capt": 15,
    "Mme": 16,
    "Sir": 17
}
yerler = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "T": 8
}
print("%10")
egitimverisi["Sex"] = egitimverisi["Sex"].map({"male": 1, "female": 0})
egitimverisi["Title"] = egitimverisi["Name"].str.extract(" ([A-Za-z]+)", expand = False)
egitimverisi["Title"] = egitimverisi["Title"].map(unvanlar)
print("%40")
egitimverisi["Title"] = egitimverisi["Title"].fillna(0).astype(int)
egitimverisi["Age"] = egitimverisi.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
print("%60")
egitimverisi = egitimverisi.drop(["Embarked", "Fare", "Ticket"], axis = 1)
egitimverisi["Cabin"] = egitimverisi["Cabin"].str.extract("([A-Z])", expand=False)
print("%80")
egitimverisi["Cabin"] = egitimverisi["Cabin"].map(yerler)
egitimverisi["Cabin"] = egitimverisi["Cabin"].fillna(0).astype(int)
val = egitimverisi.sample(frac=0.2, random_state=42)
egitimverisi = egitimverisi.drop(val.index)

print("Veri Hazır!")

def veriicindebug():
    print(f"Veri okundu. eğitim verisi satır ve sütun olarak: {egitimverisi.shape} test verisi ise: {testverisi.shape}")
    print("--- Train Verisinin İlk 5 Satırı (Feature'lar) ---")
    print(egitimverisi.head())

print("Veri dönüştürülüyor...")
x = egitimverisi.drop(["Survived", "Name", "PassengerId"], axis="columns")
y = egitimverisi["Survived"]
valx = val.drop(["Survived", "Name", "PassengerId"], axis="columns")
valy = val["Survived"]
print("%30")
x = torch.tensor(x.values, dtype=torch.float32)
xamaval = torch.tensor(valx.values, dtype=torch.float32)
yamaval = torch.tensor(valy.values, dtype=torch.float32).unsqueeze(1)
print("%50")
y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(x, y)
print("%80")
generator = torch.Generator()
generator.manual_seed(trainseed)
loader = DataLoader(dataset, batch_size=16, shuffle=True ) #,generator=generator
print("Veri hazır!")
print("Model hazırlanıyor...")

class muhtisimmodel(siniragi.Module):
    def __init__(self, neuroncountin, neuroncountout, katmansayisi):
        super().__init__()
        self.katmanlar = siniragi.ModuleList()
        for i in range(katmansayisi):
            self.katmanlar.append(siniragi.Linear(neuroncountin,neuroncountout))
            if i == katmansayisi -1:
                break
            else:
                neuroncountin = neuroncountout
                neuroncountout = neuroncountout * 2
        for i in range(katmansayisi):
            if i == katmansayisi -1:
                neuroncountin = neuroncountout
                neuroncountout = 1
            else:
                neuroncountin = neuroncountout
                neuroncountout //= 2
            self.katmanlar.append(siniragi.Linear(neuroncountin, neuroncountout))


    def forward(self, x):
        for katmannum, katman in enumerate(self.katmanlar):
            x = katman(x)
            if katmannum != len(self.katmanlar) -1: x = torch.relu(x)
        return x

print("%30")
katastrofe2 = muhtisimmodel(7,16, 1)
print("%50")

#optimizer = torch.optim.SGD(katastrofe2.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(katastrofe2.parameters(), lr=1e-3)
tahmin = katastrofe2(x)
losshesaplayici = siniragi.BCEWithLogitsLoss()
loss = losshesaplayici(tahmin, y)

print("%80")

print("Model hazır!")
print("Model istatikleri:")
print("Model ismi: Katastrofe v2.0")
print("Model parametre sayısı:")
print(sum(p.numel() for p in katastrofe2.parameters()))

def train(katastrofe2, loader, debug=True):
    losslar = []
    for i in range(150):
        egitimtamammi = wakywakyitstimeforval(katastrofe2, debug)
        if egitimtamammi:
            for x_batch, y_batch in loader:

                optimizer.zero_grad()

                tahmin = katastrofe2(x_batch)

                loss = losshesaplayici(tahmin, y_batch)
                losslar.append(loss.item())

                loss.backward()

                optimizer.step()
            tamlosslar = sum(losslar) / len(losslar)
            losslar = []
            if debug == True: print(f"Epoch {i} ortalama loss: {tamlosslar}")
        else:
            break

def isabetorani(katastrofe2):
    with torch.no_grad():
        tahmin = katastrofe2(x)
        tahminler = (torch.sigmoid(tahmin) >= 0.5).float()
        toplam = y.numel()
        dogru = (tahminler == y).sum()
        oran = dogru / toplam
        print(f"Doğruluk: %{oran.item() * 100:.2f}")


def wakywakyitstimeforval(katastrofe2, debug=True):
    with torch.no_grad():
        tahmin = katastrofe2(xamaval)
        tahminler = (torch.sigmoid(tahmin) >= 0.5).float()
        toplam = yamaval.numel()
        dogru = (tahminler == yamaval).sum()
        oran = dogru / toplam
        print(f"Doğruluk: %{oran.item() * 100:.2f}")
        if debug:
            if oran * 100 >= 82:
                torch.save(katastrofe2.state_dict(),"katastrofe82igecti.pth")
                submitolusturmatest(testverisi, katastrofe2)
                return False
            else: return True


def submitolusturmatest(testverisi, katastrofe2):
    print("Asıl test başlıyor...")
    print("%10")
    testverisi["Sex"] = testverisi["Sex"].map({"male": 1, "female": 0})
    testverisi["Title"] = testverisi["Name"].str.extract(" ([A-Za-z]+)", expand=False)
    testverisi["Title"] = testverisi["Title"].map(unvanlar)
    passengerid = testverisi["PassengerId"].values
    print("%40")
    testverisi["Title"] = testverisi["Title"].fillna(0).astype(int)
    testverisi["Age"] = testverisi.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
    print("%60")
    testverisi = testverisi.drop(["Embarked", "Fare", "Ticket"], axis=1)
    testverisi["Cabin"] = testverisi["Cabin"].str.extract("([A-Z])", expand=False)
    print("%80")
    testverisi["Cabin"] = testverisi["Cabin"].map(yerler)
    testverisi["Cabin"] = testverisi["Cabin"].fillna(0).astype(int)
    print("Veri Hazır!")
    print("Veri dönüştürülüyor...")
    x = testverisi.drop(["Name", "PassengerId"], axis="columns")

    x = torch.tensor(x.values, dtype=torch.float32)

    with torch.no_grad():
        tahmin = katastrofe2(x)
        tahminler = (torch.sigmoid(tahmin) >= 0.5).int()

    sonuc = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": tahminler.squeeze(1).numpy()
    })

    sonuc.to_csv("sonuc.csv", index=False)

    print("sonuc.csv oluşturuldu!")

def deneme():
    debugdenemesifln = False
    seed = 0
    for i in range(0, 100):

        seed = i
        torch.manual_seed(seed)
        generator = torch.Generator()
        print(seed)
        #for de in range(0, 50):
            #trainseed = de
            #generator = torch.Generator()
            #print(trainseed)
            #generator.manual_seed(trainseed)
            #loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=generator)
        katastrofe2 = muhtisimmodel(7,16,1)
        optimizer = torch.optim.AdamW(katastrofe2.parameters(), lr=1e-3)
        train(katastrofe2, loader, debug=False)
        wakywakyitstimeforval(katastrofe2,debug=False)


if __name__ == "__main__":
    if mod == "normal":
        train(katastrofe2, loader, debug=True)
    else:
        deneme()