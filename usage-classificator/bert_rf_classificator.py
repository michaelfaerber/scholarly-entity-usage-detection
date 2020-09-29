import argparse

import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel, AutoConfig

import annotation_data
import lightning_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="annotations_method.csv"
    )
    parser.add_argument(
        "--with_context",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--with_section_names",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--validation_annotation_file",
        type=str,
        default="annotations_method.csv"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # Load pre-trained model (weights)
    config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True)
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
    device = torch.device('cuda')
    model.to(device)

    # Load the annotation features and labels from the .csv file
    df = annotation_data.load_annotation_data(args.annotation_file)
    # Encode sentences into BERT embeddings
    features = annotation_data.get_bert_embeddings(model, df, args.with_context, args.with_section_names)
    X_embeddings = [f.input_tokens[0] for f in features]
    y = [f.label for f in features]
    # Split into train and test data set
    X_train, X_test, y_train, y_test = annotation_data.get_train_test_data(X_embeddings, y)

    classifier = RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=42)
    classifier.fit(X_train, y_train)

    if args.validation_annotation_file != args.annotation_file:
        # Load the annotation features and labels from the .csv file
        df = annotation_data.load_annotation_data(args.validation_annotation_file)
        # Encode sentences into BERT embeddings
        features = annotation_data.get_bert_embeddings(model, df, args.with_context)
        X_embeddings = [f.input_tokens[0] for f in features]
        y = [f.label for f in features]
        # Split into train and test data set
        _, X_test, _, y_test = annotation_data.get_train_test_data(X_embeddings, y)

    y_pred = classifier.predict(X_test)
    print(lightning_utils.acc_and_f1(y_pred, y_test))
