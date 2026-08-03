import torch
import torch.nn as siniragi

x = torch.tensor([[1.0]])
nihaihedef = torch.tensor([[17.0]])
losshesaplayici = siniragi.MSELoss()
neuroncountin = 1
neuroncountout = 4
katmansayisi = 1

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

model = muhtisimmodel(1,4, katmansayisi)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


print("Model eğitimi başlasın!")
print("Eğitim istatistikleri")
print("Hedefimiz 5 ")
print("Model parametre sayısı:")
print(sum(p.numel() for p in model.parameters()))
performans = "Kötü"
def train(katmansayisi, performans, model, optimizer, neuroncountin, neuroncountout):
    losslar = []
    while performans != "Güzel":
        for i in range(1, 10):
            tahmin = model(x)
            loss = losshesaplayici(tahmin, nihaihedef)
            loss.backward()
            losslar.append(loss)
            optimizer.step()
            optimizer.zero_grad()
            print("===============================================")
            print(f"Hedef: 5 Model ne yaptı:{tahmin} ")
            print(f"epoch{i}, loss: {loss}")
            if i in [1, 2]: pass
            else:
                optimizer.param_groups[0]['lr'] -= 0.00001
        tamlosslar = sum(losslar[-5:]) /5
        if tamlosslar <= 2: performans = "Güzel"
        else: performans = "Kötü"
        print(f"Bu modelin performansı:{performans} çünkü {tamlosslar}")
        katmansayisi += 1
        model = muhtisimmodel(1,4, katmansayisi)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tamlosslar = 0
        losslar = []
        neuroncountin = 1
        neuroncountout = 4

if __name__ == "__main__":
    train(katmansayisi, performans, model, optimizer, neuroncountin, neuroncountout)
