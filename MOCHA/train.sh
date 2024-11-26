
################################## Train master policy ##################################
cd MasterPolicy
export ALFRED_ROOT=$(pwd)
ln -s ../env .
ln -s ../gen .
ln -s ../exp .
ln -s ../data .
ln -s ../autoaugment.py .
python models/train/train_seq2seq.py \
				--dout exp/MasterPolicy \
				--batch 16 \
				--lr 1e-4 \
				--gpu \
				--save_every_epoch \
				--panoramic \
				--panoramic_concat 
cd ..
#########################################################################################


############################# Training interaction policies #############################
cd Interactions
export ALFRED_ROOT=$(pwd)
ln -s ../env .
ln -s ../gen .
ln -s ../exp .
ln -s ../data .
ln -s ../autoaugment.py .
subgoals=(CleanObject HeatObject CoolObject SliceObject ToggleObject PickupObject PutObject)
# Loop through each subgoal and execute the training command
for subgoal in "${subgoals[@]}"; do
    python models/train/train_seq2seq.py \
        --subgoal_analysis=${subgoal} \
        --dout exp/${subgoal} \
        --batch 16 \
        --lr 1e-3 \
        --gpu \
        --save_every_epoch
done
cd ..
#########################################################################################


########################## Train policy composition controller ##########################
cd PCC
export ALFRED_ROOT=$(pwd)
ln -s ../env .
ln -s ../gen .
ln -s ../exp .
ln -s ../data .
ln -s ../autoaugment.py .
python models/train/train_seq2seq.py \
				--dout exp/PCC \
				--batch 16 \
				--lr 1e-3 \
				--gpu \
				--save_every_epoch \
				--panoramic \
				--panoramic_concat 
cd ..
#########################################################################################


############################## Train Object Encoding Module #############################
cd OEM
export ALFRED_ROOT=$(pwd)
ln -s ../env .
ln -s ../gen .
ln -s ../exp .
ln -s ../data .
ln -s ../autoaugment.py .
python models/train/train_seq2seq.py \
				--dout exp/OEM \
				--batch 16 \
				--lr 1e-3 \
				--gpu \
				--save_every_epoch \
				--panoramic \
				--panoramic_concat
cd ..
#########################################################################################

