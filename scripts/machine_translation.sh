time=`date +%y-%m-%d-%T`

local_rank=1
train=true 
test=false 
model='T5'
tokenizer='t5-base'
TASK='machine_translation'
model_path='models/save/'
dataset_path='multi30k'
batch_size=16


for string in ${TASK} ${model} ${data_path} ${time}; do
    model_path+=${string}
    model_path+='_'
done
model_path+='.pt'
echo ${model_path}



if ${train} -eq true; then
    CUDA_VISIBLE_DEVICES=${local_rank} python3 run.py --training --task ${TASK} --model ${model} --tokenizer ${tokenizer} --dataset_path ${dataset_path} --model_path ${model_path}\
    > results/train/${TASK}_${model}_${dataset_path}_${time}.log
fi

if ${test} -eq true; then
    CUDA_VISIBLE_DEVICES=${local_rank} python3 run.py --testing --task ${TASK} --model ${model} --tokenizer ${tokenizer} --dataset_path ${dataset_path}  --model_path ${model_path}\
                                                    --batch_size ${batch_size}
    > results/test/${TASK}_${model}_${dataset_path}_${time}.log 
fi