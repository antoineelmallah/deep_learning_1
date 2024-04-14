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
        batch_size=32,
        shuffle=True,
        num_workers=4,
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

### EXECUCAO: ###

(cifar_classes, data_train, data_test) = carregar_dados()
# inspecionar_instancias()
(data_train, data_test) = pre_processamento()
# visualizar_imagens()
(loader_train, loader_test) = criar_dataloader()