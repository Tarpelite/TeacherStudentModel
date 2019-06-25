# Installation
```
cd TeacherStudentModel/examples/BERT

sudo pip install .
```
# Run TeacherStudentModel
TeacherStudentModel

```
python BERT\examples\TeacherStudentModel_for_dbpedia.py \
--task_name dbpedia \
--do_train --do_eval --do_lower_case \
--data_dir {data_dir} \
--bert_model {bert_model} \
--max_seq_length 512 --train_batch_size 8 \
--learning_rate 2e-5 --num_train_epochs 10.0 \
--output_dir {output_dir}
```
# data and pretrained_model


```
--task_name dbpedia \ 
--do_train --do_eval --do_lower_case --do_balance\ 
--data_dir F:\shaohanh\git\PFC\raw_data\dbpedia\few_shot \
--bert_model F:\shaohanh\git\pytorch-pretrained-BERT\data\bert-base-uncased \
--max_seq_length 128 --train_batch_size 4 \
--eval_batch_size 4 --learning_rate 2e-5 \
--num_train_epochs 10.0 \
--num_student_train_epochs 3.0 \
--top_k 500 --recurrent_times 2 \
--output_dir F:\shaohanh\git\PFC\raw_data\dbpedia\few_shot\ts_model \


```