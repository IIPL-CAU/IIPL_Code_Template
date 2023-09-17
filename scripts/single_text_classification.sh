time=`date +%y-%m-%d-%T`
train=true 
test=true 
model='bert-base-uncased'
tokenizer='bert-base-uncased'
TASK='single_text_classification'
model_path='models/save/'
dataset_path='imdb'



for string in ${TASK} ${model} ${data_path} ${time}; do
    model_path+=${string}
    model_path+='_'
done
model_path+='.pt'
echo ${model_path}



if ${train} -eq true; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --training --task ${TASK} --model ${model} --tokenizer ${tokenizer} --dataset_path ${dataset_path} --model_path ${model_path}\
    > results/train/${TASK}_${model}_${dataset_path}_${time}.log
fi

if ${test} -eq true; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --testing --task ${TASK} --model ${model} --tokenizer ${tokenizer} --dataset_path ${dataset_path}  --model_path ${model_path}\
    > results/test/${TASK}_${model}_${dataset_path}_${time}.log 
fi