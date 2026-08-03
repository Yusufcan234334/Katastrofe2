import torch
import torch.nn as siniragi
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score

egitimverisi = pd.read_csv("titianicdatas/train.csv")
testverisi = pd.read_csv("titianicdatas/test.csv")

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

print("Veri dönüştürülüyor...")
x = egitimverisi.drop(["Survived", "Name", "PassengerId"], axis="columns")
y = egitimverisi["Survived"]
valx = val.drop(["Survived", "Name", "PassengerId"], axis="columns")
valy = val["Survived"]
print("Tamam!")

model = lgb.LGBMClassifier(n_estimators=150,num_leaves=13,learning_rate=0.03,random_state=1)

model.fit(x, y)

tahminler = model.predict(valx)

skor = accuracy_score(valy, tahminler)
print(skor)