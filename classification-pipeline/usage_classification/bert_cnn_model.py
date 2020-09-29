import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from transformers import AutoConfig, AutoTokenizer, AutoModel

from usage_classification.base import BaseTransformer


class KimCNN(BaseTransformer):
    """
    From: https://romanorac.github.io/machine/learning/2019/12/02/identifying-hate-speech-with-bert-and-cnn.html

    Steps of KimCNN:
    1. Take a word embedding on the input [n x m], where n represents the maximum number of words in a sentence and
    m represents the length of the embedding.
    2. Apply convolution operations on embeddings. It uses multiple convolutions of different sizes [2 x m], [3 x m]
    and [4 x m]. The intuition behind this is to model combinations of 2 words, 3 words, etc. Note, that convolution
    width is m - the size of the embedding. This is different from CNNs for images as they use square convolutions
    like [5 x 5]. This is because [1 x m] represents a whole word and it doesn't make sense to run a convolution
    with a smaller kernel size (eg. a convolution on half of the word).
    3. Apply Rectified Linear Unit (ReLU) to add the ability to model nonlinear problems.
    4. Apply 1-max pooling to down-sample the input representation and to help to prevent overfitting. Fewer
    parameters also reduce computational cost.
    5. Concatenate vectors from previous operations to a single vector.
    6. Add a dropout layer to deal with overfitting.
    7. Apply a softmax function to distribute the probability between classes. Our network differs here because we
    are dealing with a multilabel classification problem - each comment can have multiple labels (or none). We use a
    sigmoid function, which scales logits between 0 and 1 for each class. This means that multiple classes can be
        predicted at the same time.
    """

    def __init__(self, hparams):
        """
        :param hparams: hyper parameters.
        :param embed_num: represents the maximum number of words in a sentence.
        :param embed_dim: represents the size of BERT embedding (768).
        :param class_num: is the number of labels (2, used and not-used).
        :param kernel_num: is the number of filters for each convolution operation (eg. 3 filters for [2 x m]
        convolution)
        :param kernel_sizes: of convolutions. Eg. look at combinations 2 words, 3 words, etc.
        :param dropout: is the percentage of randomly set hidden units to 0 at each update of the training phase.
        :param static: True means that we don't calculate gradients of embeddings and they stay static. If we set it to
        False, it would increase the number of parameters the model needs to learn and it could overfit.
        """
        super(KimCNN, self).__init__(hparams)

        V = self.hparams.embed_num
        D = self.hparams.embed_dim
        C = self.hparams.class_num
        Co = self.hparams.kernel_num
        Ks = self.hparams.kernel_sizes

        self.static = self.hparams.static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output


config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True)
bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
device = torch.device('cuda')
bert_model.to(device)
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = KimCNN.load_from_checkpoint("models/bert_cnn_nocontext.ckpt")
model.to(device)


def usage_classification(ner, sentence, pre_sentence, post_sentence, section_name) -> float:
    """
    Classifies a single entity as used or not used.
    :param model:
    :param ner:
    :param sentence:
    :param pre_sentence:
    :param post_sentence:
    :param section_name:
    :return: probability for used
    """
    encoded_text = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors="pt",
        pad_to_max_length=True,
        max_length=512,
        truncation=True,
        truncation_strategy="only_second",
        return_attention_mask=True
    )
    encoded_text = {name: tensor.to(device) for name, tensor in encoded_text.items()}
    with torch.no_grad():
        preds = bert_model(**encoded_text)[0]
        preds = model(preds)[0]

    return F.softmax(preds, dim=0)[1].cpu().item()
