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

need to add by yourself
