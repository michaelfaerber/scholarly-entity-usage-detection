# @author Daniel Vliegenthart

import os
from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities, filtering
from config import ROOTPATH, data_date

model_names = ['dataset_50', 'method_50']

# iteration = 'coner_' + data_date
filter_iteration = 0
expansion_iteration = 1

def main():
    
  # Generate data statistics for filtering
  print("\n\n#################################")
  print("##### FILTERINGS STATISTICS #####")
  print("#################################")
  
  for model_name in [model_names[1]]:
    rel_scores, coner_entity_list = filtering.read_coner_overview(model_name, data_date)
    filter_results = []

    nr_entities = len(read_extracted_entities(model_name, filter_iteration))

    filter_results.append(['Pointwise Mutual Information', execute_filter(model_name, 'pmi', filter_iteration)])

    filter_results.append(['Wordnet + Stopwords', execute_filter(model_name, 'ws', filter_iteration)])

    filter_results.append(['Similar Terms', execute_filter(model_name, 'st', filter_iteration)])

    filter_results.append(['Knowledge Base Look-up', execute_filter(model_name, 'kbl', filter_iteration)])

    filter_results.append(['Ensemble Majority Vote', execute_filter(model_name, 'majority', filter_iteration)])

    filter_results.append(['Coner Human Feedback', execute_filter(model_name, 'coner', filter_iteration)])

    filter_results.append(['Coner Human Feedback + Ensemble Majority Vote', execute_filter(model_name, 'mv_coner', filter_iteration)])

    print(f'{model_name}: Entities evaluated by Coner: {len(rel_scores.keys())}')
    print(f'{model_name}: Extracted entities evaluated: {nr_entities}')


    # Overview of ratings for facets and categories
    print(f'\n\n<MODEL NAME>: <FILTERING METHOD> filter kept <FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>) of unfiltered extracted entities by model\n-------------------------------------------------------------------------------------------------------')

    header = [f'<MODEL NAME>', '<FILTERING METHOD>', f'<FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>)']
    print("{: <20} {: <50} {: <40}".format(*header))

    table_data = []
    for results in filter_results:
      if results is None: continue
      table_data.append([model_name, results[0], f'{len(results[1])}/{nr_entities} ({round(float(len(results[1]*100))/nr_entities,1)}%)'])

    for row in table_data:
      print("{: <20} {: <50} {: <40}".format(*row))

  # Generate data statistics for expansion
  print("\n\n################################")
  print("##### EXPANSION STATISTICS #####")
  print("################################")
  
  for model_name in [model_names[1]]:
    rel_scores = term_sentence_expansion.read_coner_overview(model_name, data_date)
    nr_added_entities = len(rel_scores.keys())
    expansion_results = []
    nr_seeds = len(read_seeds(model_name, expansion_iteration))

    expansion_results.append(['Term Expansion', execute_expansion(model_name, 'te', expansion_iteration)])
    expansion_results.append(['Term Expansion + Coner Expansion', execute_expansion(model_name, 'tece', expansion_iteration)])
    expansion_results.append(['Term Expansion + Coner Expansion (Separate Clustering)', execute_expansion(model_name, 'tecesc', expansion_iteration)])

    print(f'{model_name}: Extracted entities evaluated: {nr_entities}')
    print(f'{model_name}: Coner entities of type "selected" and rated as "relevant: {nr_added_entities}')

    # Overview of ratings for facets and categories
    print(f'\n\n<MODEL NAME>: <EXPANSION METHOD> expanded entities from <SEED ENTITIES> to <EXPANDED ENTITIES> (<PERCENTAGE>)\n-------------------------------------------------------------------------------------------------------')

    header = [f'<MODEL NAME>', '<EXPANSION METHOD>', f'<SEED ENTITIES> -> <EXPANDED ENTITIES> (<PERCENTAGE>)']
    print("{: <20} {: <60} {: <40}".format(*header))

    table_data = []
    for results in expansion_results:
      if results is None: continue
      table_data.append([model_name, results[0], f'{nr_seeds} -> {nr_seeds + len(results[1])} ({round(float((nr_seeds + len(results[1]))*100)/nr_seeds,1)}%)'])

    for row in table_data:
      print("{: <20} {: <60} {: <40}".format(*row))

def execute_filter(model_name, filter_name, iteration):
  context_words = { 'dataset_50': ['dataset', 'corpus', 'collection', 'repository', 'benchmark'], 'method_50': ['method', 'algorithm', 'approach', 'evaluate'] }
  original_seeds = { 'dataset_50': ['buzzfeed', 'pslnl', 'dailymed', 'robust04', 'scovo', 'ask.com', 'cacm', 'stanford large network dataset', 
    'mediaeval', 'lexvo', 'spambase', 'shop.com', 'orkut', 'jnlpba', 'cyworld', 'citebase', 'blog06', 'worldcat', 
    'booking.com', 'semeval', 'imagenet', 'nasdaq', 'brightkite', 'movierating', 'webkb', 'ionosphere', 'moviepilot', 
    'duc2001', 'datahub', 'cifar', 'tdt', 'refseq', 'stack overflow', 'wikiwars', 'blogpulse', 'ws-353', 'gerbil', 
    'wikia', 'reddit', 'ldoce', 'kitti dataset', 'specweb', 'fedweb', 'wt2g', 'as3ap', 'friendfeed', 'new york times', 
    'chemid', 'imageclef', 'newegg'], 'method_50': ['hierarchical agglomerative', 'selection algorithm', 'stochastic gradient descent', 
    'pearson correlation', 'semantic relevance', 'gpbased', 'pattern matching', 'clir', 'random forest', 'random indexing', 'basic load control method', 
    'linear regression', 'recursive function', 'latent dirichlet allocation', 'convolutional dnn', 'likelihood function', 'folding-in', 'restricted boltzmann machine', 
    'lstm', 'radial basis function network', 'bmecat', 'lib', 'fast fourier', 'adaptive filter', 'spectral clustering', 'dmp method', 'reinforcement learning', 
    'graph-based propagation', 'semantictyper', 'hierarchical clustering', 'variational em', 'qald', 'fourier analysis', 'simple random algorithm', 'random search', 
    'lsh method', 'regular expression', 'rapid7', 'word embedding', 'autoencoder', 'bayesian nonparametric', 'variational bayesian inference', 'tsa algorithm', 
    'predictive modeling', 'query optimization', 'softmax', 'ridge regularization', 'tdcm', 'support vector machine', 'mcmc']}

  path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_' + filter_name + '_' + str(iteration) + '.txt'
  if os.path.isfile(path) and not filter_name in ['majority', 'coner', 'mv_coner']:
    # print("Getting filtered entities from file")
    with open(path, "r") as f:
      return [e.strip().lower() for e in f.readlines()]
  else:
    # print("Calculating filtered entities")
    if filter_name == 'pmi':
      return filtering.filter_pmi(model_name, iteration, context_words[model_name])
    if filter_name == 'ws':
      return filtering.filter_ws(model_name, iteration)
    if filter_name == 'st':
      return filtering.filter_st(model_name, iteration, original_seeds[model_name])
    if filter_name == 'kbl':
      return filtering.filter_kbl(model_name, iteration, original_seeds[model_name])
    if filter_name == 'majority':
      return filtering.majority_vote(model_name, iteration)
    if filter_name == 'coner':
      return filtering.filter_coner(model_name, iteration)
    if filter_name == 'mv_coner':
      return filtering.filter_mv_coner(model_name, iteration)

    return None

def execute_expansion(model_name, expansion_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_expanded_seeds_' + expansion_name + '_' + str(iteration) + '.txt'
  if os.path.isfile(path):
    # print("Getting filtered entities from file")
    with open(path, "r") as f:
      return [e.strip().lower() for e in f.readlines()]
  else:
    # print("Calculating filtered entities")
    if expansion_name == 'te':
      return term_sentence_expansion.term_expansion(model_name, iteration)
    if expansion_name == 'tece':
      return term_sentence_expansion.coner_term_expansion(model_name, iteration)
    if expansion_name == 'tecesc':
      return term_sentence_expansion.coner_term_expansion_separate_clustering(model_name, iteration)
    
    return None

def read_extracted_entities(model_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(iteration) + '.txt'
  with open(path, "r") as f:
    extracted_entities = [e.strip().lower() for e in f.readlines()]
  f.close()
  return extracted_entities

def read_seeds(model_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(iteration) + '.txt'
  with open(path, "r") as f:
    seeds = [e.strip().lower() for e in f.readlines()]
  f.close()
  return seeds

if __name__ == "__main__":
  main()
