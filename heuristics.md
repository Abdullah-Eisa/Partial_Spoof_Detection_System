

# PartialSpoof paper


```markdown
# PartialSpoof Paper - Configuration and Hyperparameters

This document organizes the configuration and hyperparameters used in the **PartialSpoof** paper, based on the provided `config_ps.config_test_on_eval` file and training script details.

---

## Input Parameters

- **trn_set_name**: Name of the training dataset.
- **trn_lst**: Training list file.
- **trn_input_dirs**: Directories containing the training input data.
- **input_exts**: File extensions of the input data.
- **input_dims**: Dimensions of the input data.
- **input_reso**: Resolution of the input data.
- **input_norm**: Normalization applied to the input data.
  
## Output Parameters

- **output_dirs**: Directories for storing the output data.
- **output_exts**: File extensions for the output data.
- **output_dims**: Dimensions of the output data.
- **output_reso**: Resolution of the output data.
- **output_norm**: Normalization applied to the output data.

## Other Parameters

- **params**: Additional model parameters (specifics not provided).
- **truncate_seq**: Sequence truncation (value: `None`).
- **min_seq_len**: Minimum sequence length.
- **save_mean_std**: Option to save mean and standard deviation values.
- **wav_samp_rate**: Sampling rate of the waveform (value: 16000).

---

## Training Configuration

The following configuration parameters are used during training:

- **batch_size**: 8
- **epochs**: 5000
- **learning_rate**: 0.00001
- **lr_decay_factor**: 0.5 (learning rate decay every 10 epochs)
- **lr_scheduler_type**: 1 (specific type of learning rate scheduler)
- **num_workers**: 4 (for parallelized training)
- **prefetch_factor**: None (other configurations to explore)
- **optimizer**: Adam (default configuration: β1 = 0.9, β2 = 0.999, epsilon = 10^-8)
- **sampler**: block_shuffle_by_length (for shuffling data by length)
- **epochs_without_best_check**: 50 (do not save the best model every epoch)
- **save_each_epoch**: Disabled (`--not-save-each-epoch`)

---

## Training Command

```bash
if [ $stage -le 1 ]; then
    python main.py --module-model model --model-forward-with-file-name --seed 1 \
        --ssl-finetune \
        --multi-scale-active utt 64 32 16 8 4 2 \
        --num-workers 4 --epochs 5000 --no-best-epochs 50 --batch-size 8 --not-save-each-epoch \
        --sampler block_shuffle_by_length --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.00001 \
        --module-config config_ps.config_test_on_eval \
        --temp-flag ${CON_PATH}/segment_labels/train_seglab_0.01.npy \
        --temp-flag-dev ${CON_PATH}/segment_labels/dev_seglab_0.01.npy --dev-flag > ${OUTPUT_DIR}/log_train 2> ${OUTPUT_DIR}/log_err
fi
```

---

## Training Dataset and Sequence Details

- **Acoustic Feature Frames**:
    - The number of acoustic feature frames **N** is determined by the waveform length **T** and CNN encoder configuration.
    - For pre-trained SSL models, the relationship is:  
      \( N = \frac{T}{320} \)
    - With a sampling rate of 16 kHz, the frame shift is 20 ms.

---

## Training Settings Summary

- **Wav Sample Rate**: 16000 Hz
- **Truncate Sequence**: None
- **Batch Size**: 8
- **Epochs**: 5000
- **Learning Rate**: 1e-5 (decays every 10 epochs)
- **Optimizer**: Adam (β1 = 0.9, β2 = 0.999, epsilon = 1e-8)
- **Number of Workers**: 4
- **Prefetch Factor**: None
- **Sequence Truncation**: None

---

## Notes

- The maximum size in the second dimension of the tensors during the first 1600 training examples is 393.
- The Adam optimizer with default settings is used throughout the training, and the learning rate is halved every 10 epochs.
- All experiments were repeated three times with different random seeds for the CM initialization.
```
This markdown document organizes the configurations, hyperparameters, and training setup details for the PartialSpoof paper.
```



# Speech Partial Spoofing Detection Using Conformer Blocks and Multiple Pooling Integration

This document provides an overview of the architecture, hyperparameters, and optimization details used in the paper "Speech Partial Spoofing Detection Using Conformer Blocks and Multiple Pooling Integration."

## Input Data

- **Feature Type**: 60-dimensional LFCC (Log-Frequency Cepstral Coefficients)
- **FFT Configuration**: 512-point FFT
- **Windowing**: 20 ms Hanning window
- **Window Shift**: 10 ms
- **Feature Length**: Variable-length features based on the audio input

## Model Architecture

The architecture includes several components for feature extraction and classification:

1. **Feature Extraction**:
   - **LFCC**: Extracted from both real and spoofed speech.
     - **Blue**: Real speech
     - **Red**: Spoofed speech

2. **Encoder**:
   - **Model**: SELCNN (Squeeze-and-Excitation Convolutional Neural Network)
   - **Parsimony Factor**: 2 in the SE block

3. **Refinement**:
   - Convolutional layers and **Conformer Blocks** for feature refinement.

4. **Global Contextualization**:
   - **Bi-LSTM** layer extracts global context from the features.

5. **Segmentation to Final Score**:
   - The segment features pass through:
     - **Fully Connected (FC) Layer**
     - **Sigmoid Layer** to produce segment scores
   - **Utterance-level scores** are calculated using:
     - **Power Pooling**
     - **Auto-softmax Pooling**
     - **Max Pooling**
   - Final scores are derived through **weighted summation**.

## Conformer Block

The **Conformer Block** consists of the following modules:

1. **Feed-Forward Modules**:
   - **Normalization Layer**
   - **Linear Layer** with a switch activation function
   - **Second Linear Layer**
   - **Dropout** for network regularization

2. **Multi-Head Self-Attention Module**:
   - Uses **relative sinusoidal position coding** (from Transformer-XL) to enhance robustness to speech length variations.

3. **Convolution Module**:
   - **Normalization Layer**
   - **Point-by-Point Convolution Layer** with a **Gated Linear Unit (GLU)** activation
   - **1D Depthwise Convolution Layer**
   - **Batch Normalization Layer**
   - **Rapid Activation Function**
   - **Point-Directional Convolution Layer**

## Hyperparameters

- **Optimization**:
  - **Optimizer**: Adam
  - **Learning Rate**: 0.0003 (halved every 10 epochs)
  - **Batch Size**: 32
  - **Epochs**: 50
  - **Random Seed**: Values from 0 to 5 (seed value set to 10k)

- **Regularization Parameters**:
  - **λ1**: 1e-4
  - **λ2**: 1e-4

- **Pooling Integration Weights**:
  - **ω1**: 0.4
  - **ω2**: 0.4
  - **ω3**: 0.2

## Training and Evaluation

- **Training Duration**: 50 epochs
- **Results**: The results of each experiment are averaged over multiple runs, using random seeds from 0 to 5.

---

## Network Flow Overview

The process flow of the model is as follows:

1. **Input**: 
   - **LFCC Features** extracted from real and spoofed speech.
   
2. **Feature Processing**: 
   - Features are passed through **SELCNN** to extract segmentation-level features.
   
3. **Feature Refinement**: 
   - The segmentation features undergo refinement using **Convolutional Layers** and **Conformer Blocks**.
   
4. **Contextualization**:
   - The **Bi-LSTM** layer processes the refined features to capture global contextual relationships.
   
5. **Classification**:
   - Features are passed through an **FC Layer** and **Sigmoid Layer** for segment-level score generation.
   - Utterance-level scores are computed using a combination of **Pooling Methods**:
     - Power Pooling
     - Auto-softmax Pooling
     - Max Pooling
   - The final utterance score is obtained by **weighted summation**.

---

## Conclusion

The model employs advanced techniques such as Conformer Blocks, SELCNN, Bi-LSTM, and multiple pooling strategies to effectively detect partial spoofing in speech data. The integration of relative position coding and various convolutional modules ensures resilience to speech length variations, which is crucial for this task.










# Appendix

## PartialSpoof paper

Check --module-config config_ps.config_test_on_eval , for the following input:

mostly like PS , I will need
trn_set_name, \
trn_lst,
trn_input_dirs, \
input_exts, \
input_dims, \
input_reso, \
input_norm, \
output_dirs, \
output_exts, \
output_dims, \
output_reso, \
output_norm, \
params 
truncate_seq  
min_seq_len 
save_mean_std
wav_samp_rate 

### check possible num_workers & prefetch_factor other than  num_workers=0, prefetch_factor=None for parallelized training


#stage 1:
if [ $stage -le 1 ]; then
    python main.py --module-model model --model-forward-with-file-name --seed 1 \
	--ssl-finetune \
	--multi-scale-active utt 64 32 16 8 4 2 \
	--num-workers 4 --epochs 5000 --no-best-epochs 50 --batch-size 8 --not-save-each-epoch\
       	--sampler block_shuffle_by_length --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.00001 \
	--module-config config_ps.config_test_on_eval \
	--temp-flag ${CON_PATH}/segment_labels/train_seglab_0.01.npy \
	--temp-flag-dev ${CON_PATH}/segment_labels/dev_seglab_0.01.npy --dev-flag >  ${OUTPUT_DIR}/log_train 2> ${OUTPUT_DIR}/log_err
fi





in the first 1600 training examples:  The maximum size in the second dimension of the tensors listed is 393.


 During training, we used the Adam optimizer
with a default configuration (β1 = 0.9, β2 = 0.999,  = 10−8
).
The learning rate was initialized with 1 × 10−5
and halved
every 10 epochs.
All experiments were repeated three times
with different random seeds for CM initialization, except for
the pre-trained SSL front-end. The averaged results of the three
runs are reported

wav_samp_rate = 16000
truncate_seq = None

batch_size=8
epochs=5000
learning_rate=0.00001




Note that the number of acoustic feature frames N is
determined by the waveform length T and CNN encoder
configuration. For the pre-trained SSL models we tested, the
relationship is N = T/320. This means that, given an input
waveform with a sampling rate of 16 kHz, the extracted
acoustic features a1:N have a ‘frame shift’ of 20 ms.





## Speech Partial Spoofing Detection Using Conformer Blocks and Multiple Pooling Integration


The input data comprises variable-length
60-dimensional LFCCs, computed from a 512-point FFT with a 20 ms Hanning window
shifted by 10ms. For the encoding model, a SELCNNmodel is utilized with a parsimony
factor set to 2 in the SE block. The optimization process employs an Adam optimizer
with a learning rate of 0.0003, halved every 10 epochs. During training, a batch size
of 32 is employed, and all models are trained for 50 epochs. The results of each set
of experiments are averaged, with a seed value of 10k , where k ranges from 0 to 5.
Notably, crucial hyperparameters include λ1 and λ2, set to default values of λ1 = 10−4
and λ2 = 10−4. For multiple pooling integration, we set ω1 = 0.4, ω2 = 0.4, ω3 = 0.2.





The network’s architectural configuration is illustrated in Fig. 1. Initially, LFCC extracts
audio features from partially spoofed and real speech (blue for real, red for spoofed
in Fig. 1). These features are then processed by SELCNN to derive segmentation features,
which are further refined through convolutional layers and conformer blocks.
Next, a Bi-LSTM layer extracts global contextual relationships from all features. The
segment features pass through a fully connected (FC) layer and a sigmoid layer, producing
segment scores. Finally, utterance-level scores are calculated using power pooling,
auto-softmax pooling, and max pooling, followed by a weighted summation to obtain
the final utterance-level score.





The conformer block, illustrated in Fig. 1, is composed of three crucial modules:
two feed-forward modules, a multi-head self-attention module, and a convolution module.
Within the feed-forward module, there exists a normalization layer, a linear layer
with a switch activation function, and a subsequent linear layer. Moreover, dropout is
integrated into the feed-forward module to aid in network normalization. The multihead
self-attention module adopts the relative sinusoidal position coding scheme from
Transformer-XL [31], enhancing resilience to speech length variations through relative
position coding. The convolution module comprises a normalization layer, a point-bypoint
convolution layer featuring a gated linear unit (GLU) activation function, and a
one-dimensional depth convolution layer. Following the convolution layer are a batch
normalization layer, rapid activation function, and point-direction convolution layer.

