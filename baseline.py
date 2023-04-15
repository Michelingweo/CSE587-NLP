import torch
from torch import nn
import torch.nn.functional as F
# from Nets import *

from models import MLP
from options import args_parser
# Import necessary PyTorch modules

from torchvision.models import vit_b_16, ViT_B_16_Weights, VisionTransformer
from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer, util
import timm



args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

class ViT(nn.Module):
    def __init__(self, args, image_size):
        super(ViT, self).__init__()
        self.args = args
        if self.args.pretrain:
            self.vit = vit_b_16(pretrained=True)
            # for p in self.parameters():
            #     p.requires_grad = False
            self.classifier = MLP(1000, 768, 512, self.args.num_classes)
        else:
            self.vit = VisionTransformer(
                image_size=image_size,
                patch_size=16,
                num_heads=4,
                num_layers=2,
                mlp_dim=256,
                hidden_dim= 256,
                num_classes= self.args.num_classes)

    def forward(self, image):
        # pretrained: batch size * 224 * 224
        if self.args.pretrain:

            x = self.vit(image) # (batch size, 1000)
            output = self.classifier(x) # (batch size, num_classes)
        else: # non_pretrained: batch size * 256 * 256
            output = self.vit(image) # (batch size, num_classes)

        return output

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args
        self.config = BertConfig(hidden_size = 256, num_hidden_layers = 4, num_attention_heads = 4)
        if self.args.pretrain:

            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # for p in self.parameters():
            #     p.requires_grad = False
            self.l2 = torch.nn.Dropout(0.3)
       
            self.l3 = torch.nn.Linear(768, self.args.num_classes)

            # self.classifier = MLP(768, 512, 256, self.args.num_classes)

        else:
            self.bert = BertModel(self.config)
            self.fc = nn.Linear(256,self.args.num_classes)

    def forward(self, text, attention_mask=None, token_id=None):

        if self.args.pretrain:
            # with torch.no_grad():
            embedding = self.bert(text, attention_mask=attention_mask, token_type_ids=token_id).pooler_output
            # embedding = embedding.mean(dim=1)

            output = self.l2(embedding)
            output = self.l3(output)
        else:
            embedding = self.bert(text).pooler_output #(batch size, embedding_dim)
            output = self.fc(embedding)

        return output



class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()
        self.args = args

        self.model = timm.create_model('cspresnet50', pretrained=self.args.pretrain, num_classes=self.args.num_classes)


    def forward(self, x):
        return self.model(x)



        
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden1)
        self.fc2 = nn.Linear(dim_hidden1,dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, dim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
