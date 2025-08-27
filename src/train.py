from dataload import *
from model import *
import time
import wandb
from tqdm import tqdm

epochs = 1
learning_rate = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = VGG16(num_classes = 200).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

def train():
    wandb.init(
        entity="nymphe-shandong-agricultural-university",
        project="VGG16",
        name=time.strftime('%Y%m%d%H%M') + "_local",
        dir="../wandb",
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "dataset": "tiny-imagenet-200",
            "epochs": epochs,
        }
    )

    for epoch in range(epochs):
        print(f"———------------—— Epoch: {epoch + 1} ———------------———")
        net.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader),desc="Training Process",total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"Training loss is: {loss.item()}")
                wandb.log({"train_loss": loss.item()})

        val_correct = 0
        val_loss = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(val_loader),desc="Validation Process",total=len(val_loader)):
                data, target = data.to(device), target.to(device)
                result = net(data)
                losses = loss_fn(result, target)
                val_loss += losses.item()
                val_correct += torch.argmax(result, dim=1).eq(target).sum().item()

        total_accuracy = val_correct / len(val_loader.dataset)
        print(f"Validation loss is: {val_loss}")
        print(f"The accuracy of val dataset is:{total_accuracy}")
        wandb.log({"val_loss": val_loss, "val_accuracy": total_accuracy})

        torch.save(net.state_dict(), f"../models/VGG16_epoch_{epoch}.pth")
    wandb.finish()

train()