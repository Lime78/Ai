import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

# vad behövs? 
# lära datorn i MNIST, spara bild. skapa ett neuralNetwork class, forward class som
# berättar för ai hur den ska navigera sig. Testa bild och ai ska ge en gissning

# gråskala
img = Image.open("C:/Users/nystr/Desktop/my_digit.png").convert("L")
img = img.resize((28, 28))

toTensor = ToTensor()
img_tensor = toTensor(img)
# lägger till batchar av data så enbart inte en bild kan matas in. 1 bild, 1 färg, 28 hög 28 bred
img_tensor = img_tensor.unsqueeze(0)

# hämtar träningsdatan från MNIST
data_learn = datasets.MNIST(
    root="data",
    train=True,  # train=True eftersom vi vill att modellen ska kunna träna på detta
    download=True,
    transform=ToTensor()
)

# hämtar testdatan från MNIST
data_test = datasets.MNIST(
    root="data",
    train=False,  # testdatan använder vi för att mäta hur bra modellen är
    download=True,
    transform=ToTensor()
)

# delar upp datan i batchar (mindre grupper), bra för effektiv träning
train_loader = DataLoader(data_learn, batch_size=64, shuffle=True)
test_loader = DataLoader(data_test, batch_size=64, shuffle=False)

# själva neurala nätverket
class NeuralNetwork(nn.Module):
    def __init__(self):    
        super().__init__()
        # gör om bilden till en lång vector istället för 2D (28x28 blir 784)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # tar in 784 pixlar och skickar ut till 512 neuroner
            nn.ReLU(),              # aktiveringsfunktion (tar bort negativa värden)
            nn.Linear(512, 512),    # ännu ett dolt lager
            nn.ReLU(),
            nn.Linear(512, 10)      # output: 10 olika värden (en för varje siffra 0–9)
        )

    def forward(self, x):
        # gör om input till lång vektor
        x = self.flatten(x)
        # skickar vektorn genom nätverket
        logits = self.linear_relu_stack(x)
        return logits

# skapar modellen och bestämmer om den ska köras på GPU eller CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

# flyttar paint-bilden till rätt enhet (CPU eller GPU)
img_tensor = img_tensor.to(device)

# kollar om vi redan har en tränad modell sparad
if os.path.exists("trained_ai.pth"):
    print("Loading saved model...")
    model.load_state_dict(torch.load("trained_ai.pth"))  # laddar modellens viktdata
    model.eval()  # sätter modellen i utvärderingsläge
    skip_training = True
else:
    print("no model found, training from start...")
    skip_training = False

# skapar en loss funktion som avgör hur fel modellen gissar
loss_function = nn.CrossEntropyLoss()

# skapar en optimizer som används för att uppdatera modellens vikter i noden
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# börjar med att skapa ett visst antal ephocs (hur många gånger modellen ska se hela datasetet)
ephocs = 8  # vi använder färre nu eftersom vi sparar modellen för framtiden

# tränar bara modellen om det inte redan finns en sparad version
if not skip_training:
    for epoch in range(ephocs):
        model.train()  # sätter modellen i träningsläge
        total_loss = 0

        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                      
            loss = loss_function(outputs, labels)        
            optimizer.zero_grad()                        
            loss.backward()                              
            optimizer.step()                             
            total_loss += loss.item()

        print(f"Epoch {epoch+1} completed, Loss: {total_loss:.4f}")

    # sparar modellens vikter så du slipper träna igen nästa gång
    torch.save(model.state_dict(), "trained_ai.pth")

# oavsett om vi tränade eller laddade modellen, så vill vi alltid testa den
model.eval()
correct = 0
total = 0

# with torch.no_grad() = vi tränar inte, så vi behöver inte spara gradients
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # flyttar till rätt enhet
        output = model(images)                  # modellen gör en gissning
        predictions = torch.argmax(output, 1)   # plockar ut indexet med högsta sannolikhet (dvs modellen tror det är den siffran)
        correct += (predictions == labels).sum().item()  # räknar hur många gissningar som var rätt
        total += labels.size(0)  # lägger till hur många testbilder som ingick i batchen

# skriver ut hur säker modellen var totalt sett på testdatan
print(f"AI has {100 * correct / total:.2f}% accuracy on testdata.")

# skicka in bilden från paint till modellen
output = model(img_tensor)

# används för att få rätt siffra, och även ai gissning.
prediction = torch.argmax(output, dim=1)

print("Ai think it is : ", prediction.item())

# visar bilden + vad modellen tror det är
plt.imshow(img, cmap="gray")
plt.title(f"AI think it is: {prediction.item()}")
plt.show()
