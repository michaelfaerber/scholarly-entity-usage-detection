from typing import List, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, \
    PreTrainedModel

device = torch.device('cuda')


def init_bert_model(model_name='allenai/scibert_scivocab_uncased', finetuned_name='bert_model.bin') -> Tuple[
            PreTrainedTokenizer, PreTrainedModel]:
    """
    Initializes the BERT model and loads it into the GPU.
    :param model_name: the model name (e.g. allanai/SciBERT)
    :param finetuned_name: the path to the finetuned BERT model (trained on annotated used/not-used sentences)
    :return: Tuple[PreTrainedTokenizer, PreTrainedModel]: the BERT tokenizer and embedding model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained('models/' + finetuned_name, from_tf=False, config=config)
    model.to(device)
    return tokenizer, model


def usage_classification(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, ne: str, sentence: str,
                         pre_sentence: str, post_sentence: str, section_name: str) -> float:
    """
    Classifies a single entity as used or not used.
    :param tokenizer: a (BERT) tokenizer (returned from the init method).
    :param model: a (BERT) model (returned from the init method).
    :param ne: the named entity that should be classified (currently not used).
    :param sentence: the sentence in which the named entity appears.
    :param pre_sentence: the preceding sentence.
    :param post_sentence: the succeeding sentence.
    :param section_name: the name of the section in which the named entity appears.
    :return: the probability for used
    """
    encoded_text = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_tensors=None,
        pad_to_max_length=True,
        max_length=512,
        truncation=True,
        return_attention_mask=True,
    )
    encoded_text = {name: torch.tensor(tensor)[None, :] for name, tensor in encoded_text.items()}
    encoded_text = {name: tensor.to(device) for name, tensor in encoded_text.items()}
    with torch.no_grad():
        preds = model(**encoded_text)[0][0]

    return F.softmax(preds, dim=0)[1].cpu().item()


def batch_usage_classification(tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                               batch: List[Tuple[str, str, str, str, str]]) -> List[float]:
    """
    The preferred way for classifying multiple entities at once.
    Classifies a batch of single entities as used or not used.
    :param tokenizer: a (BERT) tokenizer (returned from the init method).
    :param model: a (BERT) model (returned from the init method).
    :param batch: the batch, containing:
            - ne: the named entity that should be classified (currently not used).
            - sentence: the sentence in which the named entity appears.
            - pre_sentence: the preceding sentence.
            - post_sentence: the succeeding sentence.
            - section_name: the name of the section in which the named entity appears.
    :return: probability for used
    """
    encoded_text = tokenizer.batch_encode_plus(
        [x[1] for x in batch],
        add_special_tokens=True,
        return_tensors=None,
        pad_to_max_length=True,
        max_length=512,
        truncation=True,
        return_attention_mask=True,
    )
    encoded_text = {name: torch.tensor(tensor).to(device) for name, tensor in encoded_text.items()}
    with torch.no_grad():
        preds = model(**encoded_text)[0]

    softmax = F.softmax(preds, dim=1)
    return [x[0].cpu().item() for x in softmax]
