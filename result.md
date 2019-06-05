## Data
|    Dataset    |  #Points  |  # Classes | % of largest class |
|  :---------   | :------:  | :------:   | :------:           |
|  TREC-6       | 5452      | 6          | 22.9%              |
|  dbpedia-10k  | 10000     | 14         | 7.15%              |
|  yelp-10k     | 10000     | 5          | 20.03%             |

## Classification Results (1% as trainig and the rest as test)

### Task 1

#### params
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|
|:------|:-----|:-----|:-----|:------|
|128    |50    |128   |10.0  |3.0    |


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|  teacher model/original bert  | 0.015558      | 0.513232    | 0.325960  |
|student model before ft        | 0.015558      | 0.545051    | 0.303232  |
|student model after ft         | 0.015558      | 0.582525    | 0.351414  | 

### Task 2

#### params
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|
|:------|:-----|:-----|:-----|:------|
|128    |8     |8     |10.0  |3.0    |


#### results

|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
| :---------                    | :------:      | :------:    | :------:  |
| teacher model/original bert   | 0.576588      | 0.846970    | 0.434949  |
|student model before ft        | 0.665679      | 0.849091    | 0.406162  |
|student model after ft         | 0.711984      | 0.889697    | 0.428788  | 

### Task 3

#### params
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|
|:------|:-----|:-----|:-----|:------|
|128    |8     |8     |10.0  |10.0    |


#### results

|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|  teacher model/original bert  | -             | 0.846970          | -       |
|student model before ft        | -             | 0.850505          |    -    |
|student model after ft         | -             | 0.876869          |   -       | 


