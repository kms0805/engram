# Continual Learning Experiments

### Requirements
- pytorch
- transformers
- trl 
- matplotlib

### Dataset

d2p_each_dataset.json from reversal_curse repository

Train set : 1사람에 대한 description이 30문장으로 paraphrase 되어있다.

Test set : 1사람에 대한 description이 10문장으로 paraphrase 되어있다.

### Code

- sequential_tuning.py

    1~30 명에 대한 지식을 순차적으로 학습, 1명 학습이 끝날 때마다 여태까지 배운 지식에 대해서 test

- sequential_tuning_freeze.py

    lm_head / emb layer freeze 한 것


- sequential_tuning_lora.py

    lora training
------------
- task_vectors.py

    model weight arithmetic 을 위해 [task vectors repo](https://github.com/mlfoundations/task_vectors)의 코드를 변형해서 사용
 
### run

sbatch sequential_tuning.sh

