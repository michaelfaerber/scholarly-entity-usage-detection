from nltk.corpus import wordnet
from nltk.corpus import stopwords
from m1_postprocessing.filtering import normalized_entity_distance

ds_names = []
mt_names = []

dataset_path = 'data/dataset_names2.txt'
with open(dataset_path, "r") as file:
    for row in file.readlines():
        ds_names.append(row.strip())

method_path = 'data/method_names2.txt'
with open(method_path, "r") as file:
    for row in file.readlines():
        mt_names.append(row.strip())

        
def is_int_or_float(s):
    """
    return 1 for int, 2 for float, -1 for not a number
    """
    try:
        float(s)
        return 1 if s.count('.') == 0 else 2
    except ValueError:
        return -1


def filter_it(ner_word, model):
    ds_sim_90 = 0
    ds_sim_80 = 0
    ds_sim_70 = 0
    ds_sim_60 = 0
    ds_sim_50 = 0

    mt_sim_90 = 0
    mt_sim_80 = 0
    mt_sim_70 = 0
    mt_sim_60 = 0
    mt_sim_50 = 0

    ner_word = ner_word.split()

    if len(ner_word) > 1:
        filter_by_wordnet = []
        filtered_words = [word for word in ner_word if word not in stopwords.words('english')]

        for word in filtered_words:
            isint = is_int_or_float(word)
            if isint != -1:
                filtered_words.remove(word)

        for word in set(filtered_words):
            in_wordnet = 1
            inds = 0

            if not wordnet.synsets(word):
                in_wordnet = 0
                filter_by_wordnet.append(word)
                
        filter_by_wordnet = ' '.join(filter_by_wordnet)

        filtered_words = ' '.join(filtered_words)
        filtered_words = filtered_words.replace('(', '')
        filtered_words = filtered_words.replace(')', '')
        filtered_words = filtered_words.replace('[', '')
        filtered_words = filtered_words.replace(']', '')
        filtered_words = filtered_words.replace('{', '')
        filtered_words = filtered_words.replace('}', '')
        filtered_word = filtered_words.replace(',', '')
        lower_filtered_word = filtered_word.lower()
        filter_by_wordnet = filter_by_wordnet.replace(' ', '_')
        pmi_data = normalized_entity_distance(filtered_word, 'dataset')
        pmi_method = normalized_entity_distance(filtered_word, 'method')

        ds_similarity = []
        mt_similarity = []
        
        lower_filtered_word = lower_filtered_word.replace(' ', '_')
        
        for ds in ds_names:
            try:
                ds = ds.replace(' ', '_')
                similarity = model.wv.similarity(ds, lower_filtered_word)
                ds_similarity.append(similarity)
                if similarity > 0.89:
                    ds_sim_90 = 1
                elif similarity > 0.79:
                    ds_sim_80 = 1
                elif similarity > 0.69:
                    ds_sim_70 = 1
                elif similarity > 0.59:
                    ds_sim_60 = 1
                elif similarity > 0.49:
                    ds_sim_50 = 1

            except:
                pass

        for mt in mt_names:
            try:
                mt = mt.replace(' ', '_')
                similarity = model.wv.similarity(mt, lower_filtered_word)
                mt_similarity.append(similarity)
                if similarity > 0.89:
                    mt_sim_90 = 1
                elif similarity > 0.79:
                    mt_sim_80 = 1
                elif similarity > 0.69:
                    mt_sim_70 = 1
                elif similarity > 0.59:
                    mt_sim_60 = 1
                elif similarity > 0.49:
                    mt_sim_50 = 1

            except:
                pass

        try:
            mt_similarity = float(sum(mt_similarity)) / len(mt_similarity)
        except ZeroDivisionError:
            mt_similarity = 0

        try:
            ds_similarity = float(sum(ds_similarity)) / len(ds_similarity)
        except ZeroDivisionError:
            ds_similarity = 0

    else:
        ner_word = ner_word[0]
        isint = is_int_or_float(ner_word)
        if isint == -1:
            filtered_words = ner_word.replace('(', '')
            filtered_words = filtered_words.replace(')', '')
            filtered_words = filtered_words.replace('[', '')
            filtered_words = filtered_words.replace(']', '')
            filtered_words = filtered_words.replace('{', '')
            filtered_word = filtered_words.replace('}', '')
            pmi_data = normalized_entity_distance(filtered_word, 'dataset')
            pmi_method = normalized_entity_distance(filtered_word, 'method')
            ds_similarity = []
            mt_similarity = []

            for ds in ds_names:
                try:
                    ds = ds.replace(' ', '_')
                    similarity = model.wv.similarity(ds, filtered_word.lower())
                    ds_similarity.append(similarity)
                    if similarity > 0.89:
                        ds_sim_90 = 1
                    elif similarity > 0.79:
                        ds_sim_80 = 1
                    elif similarity > 0.69:
                        ds_sim_70 = 1
                    elif similarity > 0.59:
                        ds_sim_60 = 1
                    elif similarity > 0.49:
                        ds_sim_50 = 1

                except:
                    pass

            for mt in mt_names:
                try:
                    mt.replace(' ', '_')
                    similarity = model.wv.similarity(mt, filtered_word.lower())
                    mt_similarity.append(similarity)
                    if similarity > 0.89:
                        mt_sim_90 = 1
                    elif similarity > 0.79:
                        mt_sim_80 = 1
                    elif similarity > 0.69:
                        mt_sim_70 = 1
                    elif similarity > 0.59:
                        mt_sim_60 = 1
                    elif similarity > 0.49:
                        mt_sim_50 = 1

                except:
                    pass

            try:
                mt_similarity = float(sum(mt_similarity)) / len(mt_similarity)
            except ZeroDivisionError:
                mt_similarity = 0

            try:
                ds_similarity = float(sum(ds_similarity)) / len(ds_similarity)
            except ZeroDivisionError:
                ds_similarity = 0

    return (filtered_word, pmi_data, pmi_method, ds_similarity, mt_similarity,
            ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90,
            mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90)
