export ALFRED_ROOT=$(pwd)
for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/CleanObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals CleanObject \
	--skip_model_unroll_with_expert
done

for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/HeatObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals HeatObject \
	--skip_model_unroll_with_expert
done

for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/CoolObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals CoolObject \
	--skip_model_unroll_with_expert
done

for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/SliceObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals SliceObject \
	--skip_model_unroll_with_expert
done

for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/ToggleObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals ToggleObject \
	--skip_model_unroll_with_expert
done

for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/PickupObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals PickupObject \
	--skip_model_unroll_with_expert
done

for i in {0..20};
do CUDA_VISIBLE_DEVICES=3 python models/eval/eval_seq2seq.py   \
	--model_path	exp/PutObject/net_epoch_${i}.pth \
	--eval_split	valid_unseen                          \
	--model		models.model.seq2seq_im_mask        \
	--data		data/json_feat_2.1.0                \
	--gpu                                               \
	--max_step	400                                 \
	--max_fail	10                                  \
	--num_threads	1 \
	--subgoals PutObject \
	--skip_model_unroll_with_expert
done


