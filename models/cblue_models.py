import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.roformer import RoFormerPreTrainedModel, RoFormerModel
from sklearn import metrics


class SingleLabelLoss(nn.Module):

    def __init__(self, ):
        super(SingleLabelLoss, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # weight = torch.tensor([1 / 1477, 1 / 3126, 1 / 4922, 1 / 9402, 1 / 7386]).cuda(device)
        # self.loss_function = nn.CrossEntropyLoss(weight=weight)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, do_predict=True):
        logits = outputs[0]
        loss = self.loss_function(logits, labels)
        if do_predict:
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        else:
            pred = None
        return loss, pred


class MultipleLabelLoss(nn.Module):

    def __init__(self, ):
        super(MultipleLabelLoss, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # weight = torch.tensor([1 / 1477, 1 / 3126, 1 / 4922, 1 / 9402, 1 / 7386]).cuda(device)
        # self.loss_function = nn.CrossEntropyLoss(weight=weight)
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        logits = outputs[0]
        target = torch.FloatTensor(labels.cpu().float()).cuda()
        # pred = torch.softmax(logits, dim=1)
        pred = torch.sigmoid(logits)
        loss = self.loss_function(logits, target)

        return loss, pred


class RoformerEncoder(RoFormerPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.init_weights()

    def forward(self, features):
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        # ����12��+���һ��
        output_states = self.roformer(input_ids, attention_mask, return_dict=False)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token

        all_layer_idx = 2
        if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
        hidden_states = output_states[all_layer_idx]
        all_layer_embeddings = hidden_states
        features = {
            'token_embeddings': output_tokens,
            'cls_token_embeddings': cls_tokens,
            'attention_mask': attention_mask,
            'all_layer_embeddings': all_layer_embeddings,
        }

        return features


class RoFormerPooler(nn.Module):
    def __init__(self):
        super().__init__()

        self.word_embedding_dimension = 768
        self.pooling_mode_cls_token = False
        self.pooling_mode_max_tokens = False
        self.pooling_mode_mean_tokens = True
        self.pooling_mode_mean_sqrt_len_tokens = False

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector
