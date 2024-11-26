# MCR-Agent
<a href="https://bhkim94.github.io/projects/MCR-Agent"> <b> Multi-Level Compositional Reasoning for Interactive Instruction Following </b> </a>
<be>
<a href="https://www.linkedin.com/in/suvaansh-bhambri-1784bab7/"> Suvaansh Bhambri* </a>,
<a href="https://bhkim94.github.io/"> Byeonghwi Kim* </a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>
<br>
<a href="https://aaai.org/Conferences/AAAI-23/"> AAAI 2023 </a>

<b> MCR-Agent </b> (<b>M</b>ulti-Level <b>C</b>ompositional <b>R</b>easoning Agent) is a multi-level compositional approach that learns to navigate and manipulate objects in a divide-and-conquer manner for the diverse nature of the entailing task.
MCR-Agent addresses long-horizon instruction following tasks based on egocentric RGB observations and natural language instructions on the <a href="https://github.com/askforalfred/alfred">ALFRED</a> benchmark.
<br>

<img src="mcr-agent.png" alt="MCR-Agent">

## Download
### Pre-extracted features
Download the ResNet-18 features and annotation files from <a href="https://huggingface.co/datasets/byeonghwikim/abp_dataset">the Hugging Face repo</a>.
```
git clone https://huggingface.co/datasets/byeonghwikim/abp_dataset data/json_feat_2.1.0
```
### Raw RGB images, depth masks, and segmentation labels (Optional)
We provide zip files that contain raw RGB images (and depth & segmentation masks) in <a href="https://huggingface.co/datasets/byeonghwikim/abp_images">the Hugging Face repository</a>, which takes about 250GB in total.
With these images, you can extract features yourself with this <a href="https://github.com/snumprlab/abp/blob/main/models/utils/extract_resnet.py">code</a>.


## Code

### Training

To train MCR-Agent, run `train.sh` with hyper-parameters below. <br>

**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files. <br>


### Evaluation
#### Task Evaluation
First we need to evaluate the individual modules using 'test_unseen.sh' in each module folder. <br>

To evaluate MCR-Agent on ALFRED validation set, input the best model paths in `test_unseen.sh` for unseen fold and `test_seen.sh` for seen fold <br>

**Note**: All hyperparameters used for the experiments in the paper are set as default. <br>

## Acknoledgment
This work is partly supported by the NRF grant (No.2022R1A2C4002300), IITP grants (No.2020-0-01361-003, AI Graduate School Program (Yonsei University) 5%, No.2021-0-02068, AI Innovation Hub 5%, 2022-0-00077, 15%, 2022-0-00113, 15%, 2022-0-00959, 15%, 2022-0-00871, 20%, 2022-0-00951, 20%) funded by the Korea government (MSIT).
