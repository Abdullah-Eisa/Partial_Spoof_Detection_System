# PartialSpoof paper

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

## check possible num_workers & prefetch_factor other than  num_workers=0, prefetch_factor=None for parallelized training


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
