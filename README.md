# Teacher-Student Model

By Shaohan Huang, Tianyu Chen

## Approach

Teacher-Student model is a universe pipeline for semi-supervised training. It consists of the following steps:

1. Take a surpervised architecture and make 2 copies of it. Let's call the 3 model Teacher-with-labeled-data(TL), Teacher-with-unlabeled-data(TU), Student-model(S)
2. Train the TL model with labelled data,  then use the TL model to label the unlabeled data, make a set of pseudo data.
3. Choose a certain num of samples from pseudo data, then use the data to train the TU model.
4. Initialize the student model with a mix of weights of TL and TU, then shuffle the labeled data and pseudo data to fine-tune the student model.
5. The student model is the final model we get.

Teacher-Student model works well with the **BERT** architecture, we improves the bert performance in classification tasks using the generated student model. 

|    Dataset    |  #Points  |  # Classes | % of largest class | Original Bert Accuracy | Teacher-Student Model Accuracy|
|  :---------   | :------:  | :------:   | :------:           | :------:               | :--------------------------:  |
|  TREC-6       | 5452      | 6          | 22.9%              | 0.733840               |0.770698      |
|  dbpedia-10k  | 10000     | 14         | 7.15%              | 0.850303               |0.886767      |
|  yelp-10k     | 10000     | 5          | 20.03%             | 0.414949               |0.434646      |


## Implementation

Now is only the pytorch version based on huggingface BERT Architecture.


## Hyperparameters and other tuning

| Param        | Default | Description                                         |
| -----------  | -----   | ----------------------------------------------------|
| task_name    | trec    | the name of dataset, choices: trec, yelp, dbpedia   |
| do_train     | true    | set true if you want to run training pipeline       |
| do_eval      | true    | set true if evaluate                                |
| do_lower_case| true    | set true if using lower_case pretrained bert model  |
| do_balance   | true    | set true if select pseudo data based on balance distribution|
| data_dir     |  -      | specify the path of your data                               |
| bert_model    |  -     | specify the path of your model                              |
| max_seq_length| 128    | the length of the input id vector                           |
| train_batch_size| 4     | the batch size for all models in the pipeline              |
| learning_rate | 2e-5   | learning rate for all models in the pipeline                |
| num_train_epochs| 10.0  | train epochs for all teacher models                        |
| num_student_train_epochs| 3.0   | train epochs for the student model                 |
| top_k           | 100   | the num of pseudo data of each class               |
| alpha           | 0.1   | the weights of TL  in the initialization of student model  |
| ft_true         | false | set true if fine tune the student model with true data     |
| ft_pseudo       | false | set true if fine tune the studnet model with pseudo data   |
| output_dir      | -     | specify the path of the output                             |
| push_message    | true  | set true if you want to push message to your phone         |



# Installation
```
cd TeacherStudentModel/BERT

sudo pip install .
```
# Run TeacherStudentModel
TeacherStudentModel

```
python BERT\teacher_student_model\main.py \
--task_name trec \
--do_train --do_eval  --do_lower_case \
--do_balance \
--data_dir {data_dir} \
--bert_model {bert_model} \
--max_seq_length 128 --train_batch_size 4 \
--learning_rate 2e-5 --num_train_epochs 10.0 \
--num_student_train_epochs 3.0 --top_k 100 --alpha 0.2 \
--ft_both  \
--output_dir {output_dir}
```

## TODO
1. use the pipeline to take care of other nlp models such as seq2seq, LSTM.
2. make the pipeline compatible to CV models
3. Automatically update the alpha of the student model.