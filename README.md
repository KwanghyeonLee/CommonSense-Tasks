# CommonSense-Tasks

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially CommonSense task.
#### This repository summarizes following researches.

## Research list
* Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents (CAPEAM) (ICCV 2023) - Byeonghwi Kim, Jinyeon Kim, Yuyeong Kim, Cheolhong Min, Jonghyun Choi.

  * CAPEAM (Context-Aware Planning and Environment-Aware Memory) is an embodied instruction following agent that incorporates semantic context (e.g., appropriate objects to interact with) in a sequence of actions and maintains the changed spatial arrangement and states of interacted objects (e.g., location that the object has been moved to) in inferring the subsequent actions.

* Story Visualization by Online Text Augmentation with Context Memory (CMOTA) (ICCV 2023) - Daechul Ahn, Daneul Kim, Gwangmo Song, SeungHwan Kim, Dongyeop Kang, and Jonghyun Choi.

  * The proposed Context Memory and Online Text Augmentation (CMOTA), is a novel memory architecture for Bi-directional Transformers along with online text augmentation. This augmentation generates multiple pseudo-descriptions to provide additional supervision during training, enhancing the model's ability to generalize to language variations during inference.

* Online Continual Learning on Hierarchical Label Expansion (Hierarchical CL) (ICCV 2023) - Byung Hyun Lee, Okchul Jung, Jonghyun Choi, Se Young Chun.

  * Continual learning enables models to adapt to new tasks while retaining previously learned knowledge. We propose a novel approach called hierarchical label expansion (HLE) that allows networks to first learn coarse-grained classes before gradually expanding to more fine-grained classes at various hierarchical depths. Our experiments show that our proposed method, which uses hierarchy-aware pseudo-labeling and selective memory management, significantly outperforms existing approaches across all hierarchical levels and conventional continual learning setups.

* Multi-level Compositional Reasoning for Interactive Instruction Following (MOCHA) (AAAI 2023) - Suvaansh Bhambri*, Byeonghwi Kim*, Jonghyun Choi

  * This work introduce MCR-Agent, a robotic system that breaks down complex domestic tasks into manageable subgoals using a three-level action policy (high-level task planning, mid-level navigation/interaction control, and low-level object manipulation). This approach achieves better performance than existing methods while maintaining human-interpretable actions.

* ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments (ReALFREAD) (ECCV 2024) - Taewoong Kim*, Cheolhong Min*, Byeonghwi Kim, Jinyeon kim, Wonje Jeung and Jonghyun Choi

  * ReALFRED is a new benchmark that uses real-world scenes, objects, and room layouts to train robotic agents for household tasks, addressing the limitations of existing simulated environments that lack real-world applicability. The benchmark extends ALFRED with larger, more realistic 3D-captured spaces, and testing shows that existing methods perform significantly worse in these more realistic environments, highlighting the need for better solutions.

* Online Boundary-Free Continual Learning by Scheduled Data Prior (SDP) (ICLR 2023) - Hyunseo Koh, Minhyuk Seo, Jihwan Bang, Hwanjun Song, Deokki Hong, Seulki Park, Jung-Woo Ha and Jonghyun Choi

  * This work introduce a more realistic "boundary-free" approach to continual learning that handles continuously changing data distributions without explicit task boundaries, unlike traditional methods that rely on discrete task splits. Our solution includes scheduled knowledge transfer and data-driven balancing between past and present learning, outperforming existing methods across multiple benchmark datasets while using a novel information theory-based forgetting measure.

* Tuning Large Multimodal Models for Videos using Reinforcement Learning from AI Feedback (VLM-RLAIF) (ACL 2024) - Daechul Ahn, Yura Choi, Youngjae Yu, Dongyeop Kang and Jonghyun Choi

  * A new approach called VLM-RLAIF uses Reinforcement Learning from AI Feedback to improve video-text alignment in video large multimodal models (VLMMs), addressing the limitations of existing methods that struggle due to insufficient multimodal training data. The system incorporates detailed video descriptions during preference feedback generation (context-aware reward modeling) and outperforms existing approaches on various video benchmarks.


## Acknowledgements
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT).
