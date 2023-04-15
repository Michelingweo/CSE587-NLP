from options import args_parser
from baseline import *
from model import *
from transformer_networks import *


import torch
from torch import nn, Tensor
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
import torch.nn.functional as F



import transformers
import matplotlib.pyplot as plt
from data.data_utils import to_one_hot, get_oxford_flowers_102, get_cub_200_2011
from PIL import Image
from torch.cuda.amp import autocast, GradScaler


args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def test(net_g, data_loader,type='Test'):
    # testing
    net_g.eval()
    
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (image, target, text) in enumerate(data_loader):
        target_oh = to_one_hot(target,args.num_classes).to(args.device)
        target = target.to(args.device)


        if model_sign == 'bert':

            log_probs = net_g(text[0].to(args.device), text[1].to(args.device), text[2].to(args.device))

        elif model_sign == 'mvp':

            log_probs, batch_representation = net_g(
                image.to(args.device), 
                text[0].to(args.device), 
                text[1].to(args.device), 
                text[2].to(args.device)
                )
        elif model_sign == 'vit':
            log_probs = net_g(img.to(args.device)) 
        else:
            log_probs = net_g(text)
        # print(f'log_probs:{log_probs.shape}')
        # print(f'label:{target.shape}')
        test_loss += F.cross_entropy(log_probs, target_oh).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    acc = 100. * correct / len(data_loader.dataset)

    test_loss /= len(data_loader.dataset)
    print('\n {} set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(type,
        test_loss, correct, len(data_loader.dataset), acc
        ))

    return acc, test_loss






torch.manual_seed(5481)
dataset = args.dataset
model_sign = args.model #gru, bert, mvp, vit
lr = args.lr
epochs = args.epochs


print(f'dataset:{dataset}')
print(f'model:{model_sign}')
print(f'learning rate:{lr}')
print(f'epochs:{epochs}')


vocab_size = 30522# size of the vocabulary used for tokenization
embedding_dim = 128# dimensionality of the embedding layer
hidden_size = 128  # number of hidden units in the GRU layer


if dataset == 'flower':

    # 7370 train images
    train_set, train_loader = get_oxford_flowers_102(split='train_val', d_batch=args.local_bs)
    # 819 test images
    test_set, test_loader = get_oxford_flowers_102(split='test', d_batch=args.bs)

    num_class = 102

elif dataset == 'cub':
    # 8855 train images
    train_set, train_loader = get_cub_200_2011(split='train_val', d_batch=args.local_bs)
    # 2933 test images
    test_set, test_loader = get_cub_200_2011(split='test', d_batch=args.bs)
    
    num_class = 200


else:
    trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
    test_set = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=64)

    num_class = 10

print(f'train set size:{len(train_set)}')
print(f'test set size:{len(test_set)}')



if model_sign == 'mvp':
    model = MVP(args=args, loss_type='all').to(args.device)
elif model_sign == 'bert':
    model = BERT(args=args).to(args.device)
elif model_sign == 'resnet50':
    model = timm.create_model('cspresnet50', pretrained=True, num_classes=num_class)
elif model_sign == 'vit':
    model = ViT(args=args, image_size=224).to(args.device)
elif model_sign == 'joint_encoder':
    model = central_jointencoder(args)
else:
    print('can not identify the model type')


# optimizer = optim.SGD(model.parameters(), lr=args.lr*10, momentum=args.momentum)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=lr)

scheduler1 = transformers.get_linear_schedule_with_warmup(optimizer, 10, epochs)
scheduler2 = CosineAnnealingLR(optimizer, T_max=40, verbose=False)
scheduler3 = LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=0.01, total_iters=100, verbose=False)
scaler = GradScaler()



list_loss = []

model.train()
model.to(args.device)


train_acc, _ = test(model, train_loader, type='Train')
test_acc, _ = test(model, test_loader)
print(f'Initial Train Acc:{train_acc}, Test Acc:{test_acc}')


for epoch in range(epochs):
    batch_loss = []
    model.train()
    for batch_idx, (image, target, text) in enumerate(train_loader):

        target_oh = to_one_hot(target, num_class).to(args.device)

        target = target.to(args.device)

        # optimizer.zero_grad()

        if model_sign == 'bert':
            optimizer.zero_grad()

            with autocast():
                output = model(text[0].to(args.device), text[1].to(args.device), text[2].to(args.device))

                loss = F.cross_entropy(output, target_oh)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        elif model_sign == 'mvp':
            optimizer.zero_grad()
            with autocast():
                output, batch_representation = model(
                    image.to(args.device), 
                    text[0].to(args.device), 
                    text[1].to(args.device), 
                    text[2].to(args.device)
                    )
                loss = model.training_loss(output, batch_representation, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        elif model_sign == 'vit':
            
            optimizer.zero_grad()
            with autocast():
                output = model(image.to(args.device))
                loss = F.cross_entropy(output, target_oh)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            optimizer.zero_grad()

            output = model(text)

            loss = F.cross_entropy(output, target_oh)
            loss.backward()
            optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(text), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        batch_loss.append(loss.item())

    if epoch <= int(epochs * 0.1):
        print(f'Round:{epoch} - learning rate:{scheduler1.get_lr()[0]}')
        scheduler1.step()
    elif epoch <= (epochs - 100):
        print(f'Round:{epoch} - learning rate:{scheduler2.get_lr()[0]}')
        scheduler2.step()
    else:
        print(f'Round:{epoch} - learning rate:{scheduler3.get_lr()[0]}')
        scheduler3.step()
    loss_avg = sum(batch_loss) / len(batch_loss)
    print('\nTrain loss:', loss_avg)
    list_loss.append(loss_avg)

    if epoch % 30 == 0:
        train_acc, _ = test(model, train_loader, type='Train')
        test_acc, _ = test(model, test_loader)
        print(f'Round:{epoch}, Train Acc:{train_acc}, Test Acc:{test_acc}')



# plot loss
plt.figure()
plt.plot(range(len(list_loss)), list_loss)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.savefig('./save/{}_{}_{}.png'.format(args.model, args.dataset, args.epochs))

# testing

print('test on', len(test_set), 'samples')
test_acc, test_loss = test(model, test_loader)
train_acc, train_loss = test(model, train_loader, type='Train')

print(f'Train Acc:{train_acc}. Train Loss:{train_loss}')
print(f'Test Acc:{test_acc}. Test Loss:{test_loss}')

torch.save(model.state_dict(), './save/trained_model/{}_{}'.format(model_sign, args.dataset))




