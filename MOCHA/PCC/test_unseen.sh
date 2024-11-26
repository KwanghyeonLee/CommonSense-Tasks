export ALFRED_ROOT=$(pwd)
for i in {0..50};
do CUDA_VISIBLE_DEVICES=0 python models/eval/eval_seq2seq.py   \
	--model_path	exp/PCC/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1
done
