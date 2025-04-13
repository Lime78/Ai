import torch
from torch import nn
from torch.utils.data import dataloader
from torchvision import datasets, transforms
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
#vad behövs? 
#lära datorn i MNIST, spara bild. skapa ett neuralNetwork class, forward class som
#berättar för ai hur den ska navigera sig. Testa bild och ai ska ge en gissning

#gråskala
img = Image.open("C:/Users/nystr/Desktop/my_digit.png").convert("L")
img = img.resize((28,28))

toTensor = ToTensor()
img_tensor = toTensor(img)
#lägger till batchar av data så enbart inte en bild kan matas in. 1 bild, 1 färg, 28 hög 28 bred
img_tensor = img_tensor.unsqueeze(0)

data_learn = datasets.MNIST(
    #root sparar bilderna i en mapp som heter data, finns den ej skapas det automatiskt.
    root = "data",
    #train false avgör vilket dataset du vill ladda, false = 10 000 bilder, true 60 000
    train = False,
    #laddar ner datasetet automatiskt om det inte finns i en angiven mapp
    download = True,
    #omvandlar bilderna till arrayer som python då kan jobba med
    transform = ToTensor()
)
data_test = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

class NeuralNetwork(nn.Module):
    #initseringsfunktion, bestämmer hur nätverket ska se ut.
    def __init__(self):    
        super().__init__()
        #gör om bilden som är 28x28 till en lång vector som är 784 värden. Detta görs eftersom att nn.Linear bara kan ta emot en lång vector, och inte en bild.
        self.flatten = nn.Flatten()
        #här bygger vi själva modellen,alltså vilka lager det ska vara, och i vilken ordning
        self.linear_relu_stack = nn.Sequential(
            #Ett fullt kopplat lager som tar in 784 värden, en bild och ger ut 512.Man kan säga att varje pixel har en vikt som påverkar 512 ny neuroner.
            nn.Linear(28*28, 512),
            #Gör modellen icke linjär, tar bort alla negativa värden hjälper ai att lära sig mer komplexa mönster.
            nn.ReLU(),
            #Gör ett ytterligare lager som ofta hjälper till vid bildigenkänning
            nn.Linear(512,512),
            nn.ReLU(),
            #Ger 10 output värden, modellen gissar på vilken siffra genom att kolla vilket värde som är störst.
            nn.Linear(512,10)
        )
    #Den här funktionen säger hur data **flödar genom modellen**. Du behöver inte kalla den själv – PyTorch gör det när du matar in något i modellen.
    def forward(self, x):
        #gör om bilden till en lång vektor
        x = self.flatten(x)
        #skickar vektorn genom hela neurala nätverket vi skapat
        logits = self.linear_relu_stack(x)
        return logits
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
img_tensor = img_tensor.to(device)


#skicka in bilden till modellen
output = model(img_tensor)
#används för att få rätt siffra, och även ai gissning.
prediction = torch.argmax(output, dim=1)
print("Ai think it is : ", prediction.item())
plt.imshow(img, cmap="gray")
plt.title(f"Modellen tror det är: {prediction.item()}")
plt.show()

