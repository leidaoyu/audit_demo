# coding: utf-8
import torch
from models.roformer import RoFormerTokenizer, RoFormerForSequenceClassification, RoFormerConfig
import os
from functions.downloader import download_file


def get_model(pretrained_model, load_model_path, num_labels):
    config = RoFormerConfig.from_pretrained(pretrained_model, num_labels=num_labels, cache_dir=None)
    model = RoFormerForSequenceClassification.from_pretrained(pretrained_model, config=config)
    model.load_state_dict(torch.load(load_model_path))

    return model


def get_tokenizer(pretrained_model):
    return RoFormerTokenizer.from_pretrained(pretrained_model)


def encode_data(text, tokenizer):
    inputs = tokenizer.encode(text=text, max_length=512, truncation=True,
                              truncation_strategy='longest_first',
                              add_special_tokens=True, pad_to_max_length=True)
    inputs = [(inputs, [1 if x != 0 else 0 for x in inputs])]
    inputs = [torch.tensor(x) for x in zip(*inputs)]

    return inputs


def model_predict(inputs, model):
    with torch.no_grad():
        outputs = model(input_ids=inputs[0], attention_mask=inputs[1])

    softmax_prob = torch.softmax(outputs[0], dim=1)
    pred_label = torch.argmax(softmax_prob, dim=1)
    predicts = pred_label.cpu().detach().numpy()

    return predicts[0]


def predict(text, model, tokenizer):
    inputs = encode_data(text, tokenizer)
    res = model_predict(inputs, model)
    return res


name_list2 = ['合规',
              '不合规']
name_list4 = ['违禁暴恐',
              '文本色情',
              '低俗辱骂',
              '恶意推广']

if not os.path.exists('./ckpt/'): os.makedirs('./ckpt/')

if not os.path.exists('./ckpt/model2.ckpt'):
    download_file(
        'https://huggingface.co/shouldbe/auditDemo/resolve/main/ContentAudit_0_1_roformer-base/E(1)_M(f1%3D0.8476).ckpt',
        './ckpt/model2.ckpt')
if not os.path.exists('./ckpt/model4.ckpt'):
    download_file(
        'https://huggingface.co/shouldbe/auditDemo/resolve/main/ContentAudit_4_roformer-base/E(2)_M(f1%3D0.8158).ckpt',
        './ckpt/model4.ckpt')

model_2 = get_model('junnyu/roformer_chinese_base', './ckpt/model2.ckpt', 2)
model_4 = get_model('junnyu/roformer_chinese_base', './ckpt/model4.ckpt', 4)
tokenizer = get_tokenizer('junnyu/roformer_chinese_base')
