import subprocess
import re
from config import ROOTPATH
import sys


def train_model(model_name: str, training_cycle: int) -> None:
    """
    Executes the training process for the NER model
    :param model_name:
    :param training_cycle:
    """
    print('Training the model...')
    sys.stdout.flush()
    output_file = open(ROOTPATH + '/crf_trained_files/temp' + model_name + str(training_cycle) + '.txt', 'a')
    command = ('java -cp ' + ROOTPATH + '/stanford_files/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop '
               + ROOTPATH + '/prop_files/' + model_name + '_' + str(training_cycle) + '.prop')

    subprocess.call(command, stdout=output_file, stderr=subprocess.STDOUT, shell=True)


def create_prop(model_name: str, training_cycle: int, sentence_expansion: bool) -> None:
    """
    Creates the property file to train the Stanford NER model
    :param model_name:
    :param training_cycle:
    :param sentence_expansion:
    """
    print('Creating property file for Stanford NER training')
    prop_file_path = open(ROOTPATH + '/data/blank.prop', 'r')
    text = prop_file_path.read()

    if sentence_expansion:
        train_file_path = ('trainFile=' + ROOTPATH + '/processing_files/' + model_name + '_TSE_tagged_sentence_' +
                           str(training_cycle) + '.txt')
        output_file_path = ('serializeTo=' + ROOTPATH + '/crf_trained_files/' + model_name + '_TSE_model_' +
                            str(training_cycle) + '.ser.gz')
    else:
        train_file_path = ('trainFile=' + ROOTPATH + '/processing_files/' + model_name + '_TE_tagged_sentence_' +
                           str(training_cycle) + '.txt')
        output_file_path = ('serializeTo=' + ROOTPATH + '/crf_trained_files/' + model_name + '_TE_model_' +
                            str(training_cycle) + '.ser.gz')

    edited = re.sub(r'trainFile.*?txt', train_file_path, text, flags=re.DOTALL)
    edited = re.sub(r'serializeTo.*?gz', output_file_path, edited, flags=re.DOTALL)

    text_file = open(ROOTPATH + '/prop_files/' + model_name + '_' + str(training_cycle) + '.prop', 'w')
    text_file.write(edited)
    text_file.close()
    
