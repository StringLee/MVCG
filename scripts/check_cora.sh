#!bin/bash
export PYTHONUNBUFFRED=1

cd ../
  CUDA_VISIBLE_DEVICES=3 \
	python main_full_batch.py \
		--dataset cora --encoder gat --decoder mlp --seed 3 --device cuda \
		--lr 0.0005 --max_epoch 100 \
		--E_para 10 --D_para 10 \
		--loss_E_S_para 0.5 --loss_E_A_para 1 --loss_E_Z_para 250 --decoder_AS_type mean \
		--loss_D_S_para 0.5 --loss_D_A_para 0.5 \
	  --missing_rate 0.6 \
		--num_head 2
