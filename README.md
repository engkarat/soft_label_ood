# Improving the out-of-distribution detection performance using soft-label target with corrupted images.

The dataset can be downloaded at [here](https://drive.google.com/file/d/11iuht4JyRSYEsAlDAW37EkYq6PEZYTe8/view?usp=sharing). One needs to extract the folder and place in PROJECT_DIR/data such that the path PROJECT_DIR/data/datasets exists.

#### Steps for training the soft-label network

1. Train the vanilla (1st attempt) network: ```python 1_std_net.py --run_id 1 --dset_name cifar100 --net_name wideresnet```
2. Produce the accuracy list of all corruptions: ```python 2_produce_corpus.py --run_id 1 --net_name wideresnet --dset_name cifar100```
3. Train the soft-label (2nd attempt) network: ```python 3_soft_net.py --run_id 1 --net_name wideresnet --dset_name cifar100 --acc_list_path log/1_std_net/run-1/corpus_imagenet_c```
4. Evaluate the performance on classification, OOD detection, and calibration for the vanilla network: ```python 4_evaluate.py --run_id 1 --eval_exp 1 --dset_name cifar100 --net_name wideresnet```
5. Evaluate the performance on classification, OOD detection, and calibration for the soft-label network: ```python 4_evaluate.py --run_id 1 --eval_exp 3 --dset_name cifar100 --net_name wideresnet```
