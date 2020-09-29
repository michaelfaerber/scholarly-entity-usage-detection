if [ $# -lt 5 ]
then
	echo "./usage-classificator.sh <mode:train|predict> <entity_type:method|dataset> <model_type:bert|cnn|rf> <with_context:True|False> <with_section_names:True|False> [validation_entity_type:method|dataset]"
	exit
fi

annotation_file="annotations_${2}.csv"

train_cmd="--do_predict"
if [ "$1" == "train" ]; then train_cmd="${train_cmd} --do_train"; fi

data_dir="data_${2}_${3}"
if [ $# -eq 6 ]
then
  # use model that has been trained on $6 for prediction of $1
  output_dir="output_${6}_${3}"
else
  output_dir="output_${2}_${3}"
fi

if [ "$3" == "bert" ]
then
	num_epochs=10
	python_file="bert_classificator.py"
else
	num_epochs=20
	python_file="bert_cnn_classificator.py"
fi

context_cmd=""
if [ "$4" == "True" ]
then
	data_dir="${data_dir}_context"
	output_dir="${output_dir}_context"
	context_cmd=" --with_context"
fi
section_cmd=""
if [ "$5" == "True" ]
then
	data_dir="${data_dir}_sections"
	output_dir="${output_dir}_sections"
	section_cmd=" --with_section_names"
fi

if [ "$3" == "rf" ]
then
  python bert_rf_classificator.py --annotation_file "annotations_${6}.csv" $context_cmd $section_cmd --validation_annotation_file "$annotation_file"
else
  python $python_file --data_dir "$data_dir" --output_dir "$output_dir" --train_batch_size 8 --num_train_epochs $num_epochs $train_cmd $context_cmd $section_cmd --annotation_file "$annotation_file"
fi
