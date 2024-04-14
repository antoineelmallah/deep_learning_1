# bibliotecas nativas
import copy

# bibliotecas instaladas
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T # funções de pré-processamento
from torchvision.datasets import CIFAR10

# implementação local
import utils
import visualization as vis

# carregar os dados

def carregar_dados():

    cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    data_train = CIFAR10(
        root='./data',
        train=True,
        transform=T.ToTensor(),
        download=True
    )

    data_test = CIFAR10(
        root='./data',
        train=False,
        transform=T.ToTensor(),
        download=True
    )

    return cifar_classes, data_train, data_test

# Inspecionar as instâncias

def inspecionar_instancias():
    fig = vis.visualize_cifar10(data_train, samples_per_class=20, classes=None, seed=1)
    fig.suptitle('CIFAR-10 Samples')
    plt.show()

# Pre-processamento das imagens

def pre_processamento():
    mean = [.49139968, .48215827, .44653124] # médias do conjunto de treinamento para cada camada de cor (RGB)
    std = [.24703233, .24348505, .26158768]  # desvios padrão do conjunto de treinamento para cada camada de cor (RGB)

    # Transformações estocásticas
    transforms_train = T.Compose([
        T.RandomHorizontalFlip(p=.5),   # transformação aplicada em imagem
        T.RandomRotation(10),           # transformação aplicada em imagem (rotação aleatória entre -10 e 10 graus)
        T.ToTensor(),                   # transforma valores dos pixels (0 a 255) para valores entre 0 e 1
        T.Normalize(mean=mean, std=std) # transformação aplicada em tensor (transforma em distribuição normal - media 0 e desvio padrão 1)
    ])

    # Transformações determinísticas (não estocásticas)
    transforms_eval = T.Compose([
        T.ToTensor(),                   # transforma valores dos pixels (0 a 255) para valores entre 0 e 1
        T.Normalize(mean=mean, std=std) # transformação aplicada em tensor (transforma em distribuição normal - media 0 e desvio padrão 1)
    ])

    data_train = CIFAR10(
        root='./data',
        train=True,
        transform=transforms_train,
        download=True
    )

    data_test = CIFAR10(
        root='./data',
        train=False,
        transform=transforms_eval,
        download=True
    )

    return (data_train, data_test)

def visualizar_imagens():
    fig = vis.visualize_dataset(data_train, num_instances=50, max_col=10, img_size=128, seed=1)
    fig.suptitle('Preprocessing visualization')
    plt.show()

# Criação do DataLoader (possibilita a criação de várias amostras de forma paralela. 
# Qualdo chega no fim do dataset, o DataLoader embaralha o dataset e continua fornecendo 
# amostras)

def criar_dataloader():
    loader_train = DataLoader(
        data_train,
        batch_size=32,   # Sempre que chamar o loader, vai receber 32 amostras
        shuffle=True,    # Sempre que passar por todos os dados de treinamento (a cada época), embaralhar os dados
        num_workers=4,   # número de threads usado para o processamento paralelo
        pin_memory=True   
    )

    loader_test = DataLoader(
        data_test,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return (loader_train, loader_test)

# Criar um classificador linear (apenas 1 camada)
# Module representa uma camada da rede neural, no PyTorch
# Quando o PyTorch não disponibilizar a camada desejada, 
# podemos implementá-la, herdando de torch.nn.Module

class LinearClassifier(nn.Module):
    
    def __init__(self, dim_data, num_classes) -> None:
        super().__init__()
        self.layer = nn.Linear(
            in_features=dim_data,     # numero de features que entra na camada
            out_features=num_classes, # numero de features que saem da camada (classes)
            bias=False
        )

    # Executa o forward do modelo
    def forward(self, x): # x = batch_size x tensor da imagem 3 x 32 x 32
        batch_size = x.shape[0]
        x = x.view(batch_size, -1) # vetoriza o tensor
        out = self.layer(x)
        return out
    
# Definir o processo de otimização

def train(train_loader, model, num_epochs, print_freq):

    # activate training mode
    model.train()

    # pick device depending on GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('device:', device)

    # send model to device
    model = model.to(device)

    # create optimization criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=.001)

    train_loader_len = len(train_loader)

    # saving losses to plot later
    losses = []
    losses_smooth = []
    loss_smooth = 0

    # iterate num_epoch times over entire dataset
    for epoch in range(num_epochs):
        # iterate through mini-batches
        for i, (x,y) in enumerate(train_loader):
            iter_num = (epoch * train_loader_len + i + 1)

            # send mini-batch data to device
            x = x.to(device)
            y = y.to(device)

            # compute model output (forward pass) and loss
            out = model(x)
            loss = criterion(out, y)

            if iter_num == 1:
                loss_smooth = loss.item()
            else:
                # exponential movinv average of loss (easier to visualize)
                loss_smooth = .99 * loss_smooth + .01 * loss.item()

            losses.append(loss.item())
            losses_smooth.append(loss_smooth)

            # zero gradients in optimizer, perform backward pass, and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #logging during training
            if iter_num % print_freq == 0:
                run_info = (f'epoch: { epoch + 1 }/{ num_epochs } | step: [{ i + 1 }/{ train_loader_len }] | loss_smooth: { loss_smooth } | loss: { loss }')
                print(run_info)

    # return losses for plotting
    return losses, losses_smooth

# treinar o modelo linear

def treinar_modelo_linear():
    losses, losses_smooth = train(loader_train, model_linear, num_epochs=10, print_freq=1000)
    print('Training complete. Plotting loss curve and saving weights...')

    fig = vis.visualize_losses(losses, losses_smooth)
    plt.show()

    torch.save(model_linear.state_dict(), './ckpt/model_linear_cifar10.pth')

### EXECUCAO: ###

(cifar_classes, data_train, data_test) = carregar_dados()
# inspecionar_instancias()
(data_train, data_test) = pre_processamento()
# visualizar_imagens()
(loader_train, loader_test) = criar_dataloader()
model_linear = LinearClassifier(dim_data=3*32*32, num_classes=10)
# print(model_linear)
treinar_modelo_linear()

