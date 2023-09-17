time=`date +%y-%m-%d-%T`
train=false 
test=true 
model='bert-base-uncased'
tokenizer='bert-base-uncased'
TASK='single_text_classification'
model_dir='models/save/'
model_path=${model_dir}'single_text_classification_bert-base-uncased_23-09-18-01:00:27_.pt'
dataset_path='imdb'






if ${train} -eq true; then

    model_path = model_dir
    for string in ${TASK} ${model} ${data_path} ${time}; do
        model_path+=${string}
        model_path+='_'
    done
    model_path+='.pt'

    echo ${model_path}
    CUDA_VISIBLE_DEVICES=0 python3 run.py --training --task ${TASK} --model ${model} --tokenizer ${tokenizer} --dataset_path ${dataset_path} --model_path ${model_path}\
    > results/train/${TASK}_${model}_${dataset_path}_${time}.log
fi

if ${test} -eq true; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --testing --task ${TASK} --model ${model} --tokenizer ${tokenizer} --dataset_path ${dataset_path}  --model_path ${model_path}\
    > results/test/${TASK}_${model}_${dataset_path}_${time}.log 
fi