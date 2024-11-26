export ALFRED_ROOT=$(pwd)

python models/eval/eval_seq2seq.py   \
		--nav_model_path exp/MasterPolicy/net_epoch_best.pth  \
		--pickup_model_path exp/PickupObject/net_epoch_best.pth  \
		--put_model_path exp/PutObject/net_epoch_best.pth  \
		--heat_model_path exp/HeatObject/net_epoch_best.pth  \
		--cool_model_path exp/CoolObject/net_epoch_best.pth  \
		--clean_model_path exp/CleanObject/net_epoch_best.pth  \
		--toggle_model_path exp/ToggleObject/net_epoch_best.pth  \
		--slice_model_path exp/SliceObject/net_epoch_best.pth  \
		--object_model_path exp/OEM/net_epoch_best.pth  \
		--subgoal_model_path exp/PCC/net_epoch_best.pth  \
		--eval_split	valid_seen                          \
		--data		data/json_feat_2.1.0                \
		--gpu                                               \
		--max_step	400                                 \
		--max_fail	10                                  \
		--num_threads 4; 

