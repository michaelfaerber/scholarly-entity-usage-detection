import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from m1_preprocessing import term_sentence_expansion
import nltk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd


# Compare the word clustering capabilities of word2vec and BERT for our extracted sentences
# from the 10,000 abstract sample

# Word2Vec
named_entities = term_sentence_expansion.generic_named_entities('data/bert_test.txt')
print("Named entities:", len(named_entities))

with open('data/method_names.txt', 'r') as f:
    seed_entities = [x.strip() for x in f.readlines()]

named_entities += seed_entities
print("Named + seed entities:", len(named_entities))
print(named_entities)

# Create word2vec bigrams (and replace spaces with underscores)
processed_entities = []
for pp in named_entities:
    temp = pp.split(' ')
    if len(temp) > 1:
        bigram = list(nltk.bigrams(pp.split()))
        for bi in bigram:
            bi = bi[0].lower() + '_' + bi[1].lower()
            processed_entities.append(bi)
    else:
        processed_entities.append(pp)
processed_entities = [e.lower() for e in processed_entities]
processed_entities = list(set(processed_entities))
print("Processed entities:")
print(processed_entities)

# Use the word2vec model to create word embeddings
word2vec_path = 'embedding_models/word2vec-sample.model'
#df, labels_array = term_sentence_expansion.build_word_vector_matrix(word2vec_path, processed_entities, "method")
model = Word2Vec.load(word2vec_path)
print("Loaded word2vec")
vocab = list(model.wv.vocab)
print("Vocab =", len(vocab))
X = model.wv[vocab]
print("Loaded word2vec embeddings")
tsne = TSNE()
X_tsne = tsne.fit_transform(X)
print("Converted embeddings into 2d plane")
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
print("Word2Vec embeddings:")
print(df)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)
plt.savefig("word2vec.png")

#print(df)
#print(labels_array)

exit(0)

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open('data/bert_test.txt', 'r') as f:
    sentences = [x.strip() for x in f.readlines()]

all = " ".join(sentences)
sentences_splitted = nltk.sent_tokenize(all)
tokenized_text = []
segment_ids = []
for (index, sent) in enumerate(sentences_splitted):
    if index == 0:
        sent = "[CLS] " + sent
    sent += " [SEP]"
    tokens = tokenizer.tokenize(sent)
    tokenized_text += tokens
    segment_ids += [index % 2] * len(tokens)
tokenized_text = tokenized_text[:-1]
segment_ids = segment_ids[:-1]
assert(len(tokenized_text) == len(segment_ids))

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segment_ids])
#text_bert = "[CLS] " + " [SEP] ".join(sentences_splitted)
#tokenized_text = tokenizer.tokenize(text_bert)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers.
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # becase we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]


