# MeanTeacherStudent Task 2

## Idea

Init the student model use the weights of TL and TU model, and fine tune them with true data and
pseudo data.

The formula like:  
$$ W_s = \alpha \mul W_TL + (1- \alpha) \mul W_TU $$

## Data 

|    Dataset    |  #Points  |  # Classes | % of largest class |
|  :---------   | :------:  | :------:   | :------:           |
|  TREC-6       | 5452      | 6          | 22.9%              |
|  dbpedia-10k  | 10000     | 14         | 7.15%              |
|  yelp-10k     | 10000     | 5          | 20.03%             |


## Alpha = 0.1

### settings 

| Param                        | Value | Description                                                 |
| ---------------------------  | ----- | ------------------------------------------------------------|
| task_name                    | all   | the name of dataset, choices: trec, yelp, dbpedia           |
| do_train                     | true  | set true if you want to run training pipeline               |
| do_eval                      | true  | set true if evalute                                         |
| do_lower_case                | true  | set true if using lower_case pretrained bert model          |
| do_balance                   | true  | set true if select pseudo data based on balance distribution|
| data_dir                     |  -    | specify the path of your data                               |
| bert_model                   |  -    | specify the path of your model                              |
| max_seq_length               | 128   | the length of the input id vector                           |
| train_batch_size             | 4     | the batch size for all models in the pipeline               |
| learning_rate                | 2e-5  | learning rate for all models in the pipeline                |
| num_train_epochs             | 10.0  | train epochs for all teacher models                         |
| num_student_train_epochs     | 3.0   | train epochs for all student models                         |
| top_k                        | 100   | the num of pseudo data of each class                        |
| alpha                        | 0.1   | the weights of TL  in the initialization of student model   |
| ft_true                      | false | set true if fine tune the student model with true data      |
| ft_pseudo                    | false | set true if fine tune the studnet model with pseudo data    |
| output_dir                   | -     | specify the path of the output                              |
| push_message                 | true  | set true if you want to push message to your phone          |


### results (No fine-tune/ with true data/ with pseudo data/ with both)
|    Data        |  model_TL | model_TU    | model_student                        | 
|:------------   | :-------- | :-------    | :-----------                         |            
|   trec6        | 0.733840  | 0.632525    | 0.644008/0.715132/0.642526/0.646045  |       
|   dbpedia-10k  | 0.850303  | 0.856768    | 0.860000/0.887879/0.863535/0.885454  |       
|   yelp-10k     | 0.414949  | 0.386465    | 0.386970/0.352323/0.391515/0.359292  |


## Alpha = 0.2

### settings 

| Param                        | Value | Description                                                 |
| ---------------------------  | ----- | ------------------------------------------------------------|
| task_name                    | all   | the name of dataset, choices: trec, yelp, dbpedia           |
| do_train                     | true  | set true if you want to run training pipeline               |
| do_eval                      | true  | set true if evalute                                         |
| do_lower_case                | true  | set true if using lower_case pretrained bert model          |
| do_balance                   | true  | set true if select pseudo data based on balance distribution|
| data_dir                     |  -    | specify the path of your data                               |
| bert_model                   |  -    | specify the path of your model                              |
| max_seq_length               | 128   | the length of the input id vector                           |
| train_batch_size             | 4     | the batch size for all models in the pipeline               |
| learning_rate                | 2e-5  | learning rate for all models in the pipeline                |
| num_train_epochs             | 10.0  | train epochs for all teacher models                         |
| num_student_train_epochs     | 3.0   | train epochs for all student models                         |
| top_k                        | 100   | the num of pseudo data of each class                        |
| alpha                        | 0.2   | the weights of TL  in the initialization of student model   |
| ft_true                      | false | set true if fine tune the student model with true data      |
| ft_pseudo                    | false | set true if fine tune the studnet model with pseudo data    |
| output_dir                   | -     | specify the path of the output                              |
| push_message                 | true  | set true if you want to push message to your phone          |


### results (No fine-tune/ with true data/ with pseudo data/ with both)
|    Data        |  model_TL | model_TU    | model_student                        | 
|:------------   | :-------- | :-------    | :-----------                         |            
|   trec6        | 0.733840  | 0.632525    | 0.663271/0.705871/0.632895/0.674754  |       
|   dbpedia-10k  | 0.850303  | 0.856768    | 0.862727/0.873434/0.864747/0.884040  |       
|   yelp-10k     | 0.414949  | 0.386465    | 0.387272/0.369293/0.394444/0.408484  |

## Alpha = 0.3

### settings 

| Param                        | Value | Description                                                 |
| ---------------------------  | ----- | ------------------------------------------------------------|
| task_name                    | all   | the name of dataset, choices: trec, yelp, dbpedia           |
| do_train                     | true  | set true if you want to run training pipeline               |
| do_eval                      | true  | set true if evalute                                         |
| do_lower_case                | true  | set true if using lower_case pretrained bert model          |
| do_balance                   | true  | set true if select pseudo data based on balance distribution|
| data_dir                     |  -    | specify the path of your data                               |
| bert_model                   |  -    | specify the path of your model                              |
| max_seq_length               | 128   | the length of the input id vector                           |
| train_batch_size             | 4     | the batch size for all models in the pipeline               |
| learning_rate                | 2e-5  | learning rate for all models in the pipeline                |
| num_train_epochs             | 10.0  | train epochs for all teacher models                         |
| num_student_train_epochs     | 3.0   | train epochs for all student models                         |
| top_k                        | 100   | the num of pseudo data of each class                        |
| alpha                        | 0.3   | the weights of TL  in the initialization of student model   |
| ft_true                      | false | set true if fine tune the student model with true data      |
| ft_pseudo                    | false | set true if fine tune the studnet model with pseudo data    |
| output_dir                   | -     | specify the path of the output                              |
| push_message                 | true  | set true if you want to push message to your phone          |


### results (No fine-tune/ with true data/ with pseudo data/ with both)
|    Data        |  model_TL | model_TU    | model_student                        | 
|:------------   | :-------- | :-------    | :-----------                         |            
|   trec6        | 0.733840  | 0.632525    | 0.685312/0.735692/0.612521/0.7193925 |       
|   dbpedia-10k  | 0.850303  | 0.856768    | 0.864747/0.868081/0.866364/0.886768  |       
|   yelp-10k     | 0.414949  | 0.386465    | 0.388182/0.369697/0.395859/0.361515  |

## Alpha = 0.4

### settings 

| Param                        | Value | Description                                                 |
| ---------------------------  | ----- | ------------------------------------------------------------|
| task_name                    | all   | the name of dataset, choices: trec, yelp, dbpedia           |
| do_train                     | true  | set true if you want to run training pipeline               |
| do_eval                      | true  | set true if evalute                                         |
| do_lower_case                | true  | set true if using lower_case pretrained bert model          |
| do_balance                   | true  | set true if select pseudo data based on balance distribution|
| data_dir                     |  -    | specify the path of your data                               |
| bert_model                   |  -    | specify the path of your model                              |
| max_seq_length               | 128   | the length of the input id vector                           |
| train_batch_size             | 4     | the batch size for all models in the pipeline               |
| learning_rate                | 2e-5  | learning rate for all models in the pipeline                |
| num_train_epochs             | 10.0  | train epochs for all teacher models                         |
| num_student_train_epochs     | 3.0   | train epochs for all student models                         |
| top_k                        | 100   | the num of pseudo data of each class                        |
| alpha                        | 0.4   | the weights of TL  in the initialization of student model   |
| ft_true                      | false | set true if fine tune the student model with true data      |
| ft_pseudo                    | false | set true if fine tune the studnet model with pseudo data    |
| output_dir                   | -     | specify the path of the output                              |
| push_message                 | true  | set true if you want to push message to your phone          |


### results (No fine-tune/ with true data/ with pseudo data/ with both)
|    Data        |  model_TL | model_TU    | model_student                        | 
|:------------   | :-------- | :-------    | :-----------                         |            
|   trec6        | 0.733840  | 0.632525    | 0.685312/0.735692/0.612521/0.7193925 |       
|   dbpedia-10k  | 0.850303  | 0.856768    | 0.864747/0.868081/0.866364/0.886768  |       
|   yelp-10k     | 0.414949  | 0.386465    | 0.388182/0.369697/0.395859/0.361515  |


