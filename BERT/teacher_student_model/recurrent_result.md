# Acccuracy with recurrent model

## dbpedia

### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs |student_train_epochs|top_k|Recurrent Iter|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**500**|5|

### results
|    Iter   |  teacher model  | student model 1 |  student model 2|
|  :--------| :------:        | :------:        | :------:        |
|  1        | 0.886667        | 0.908384        | 0.898283        |
|  2        | 0.898283        | 0.908081        | 0.875556        |
|  3        | 0.875556        | 0.903838        | 0.898485        |
|  4        | 0.898485        | 0.907576        | 0.888687        |
|  5        | 0.888687        | 0.898990        | 0.903131        | 


## trec

### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs |student_train_epochs|top_k|Recurrent Iter|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|:-----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**500**|5|

### results
|    Iter   |  teacher model  | student model 1 |  student model 2|
|  :--------| :------:        | :------:        | :------:        |
|  1        | 0.632339        | 0.729580        | 0.739026        |
|  2        | 0.739026        | 0.724208        | 0.778477        |
|  3        | 0.778477        | 0.722356        | 0.728283        |
|  4        | 0.728283        | 0.731802        | 0.756251        |
|  5        | 0.756251        | 0.733099        | 0.763475        | 

# yelp

### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs |student_train_epochs|top_k|Recurrent Iter|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|:-----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**500**|5|

### results
|    Iter   |  teacher model  | student model 1 |  student model 2|
|  :--------| :------:        | :------:        | :------:        |
|  1        | 0.313232        | 0.325960        | 0.412222        |
|  2        | 0.412222        | 0.344040        | 0.411818        |
|  3        | 0.411818        | 0.335253        | 0.408081        |
|  4        | 0.408081        | 0.354646        | 0.399293        |
|  5        | 0.399293        | 0.345657        | 0.419292        | 

