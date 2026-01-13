name=final_model
CUDA_VISIBLE_DEVICES=0 SLURM_NTASKS=1   python test_model.py \
        --eval_data  vulfix_data/test.json \
        --write_results \
        --model_path checkpoint_final/final_model/checkpoint/best_dev/ \
        --per_gpu_eval_batch_size 1 \
        --n_context 10 \
	      --beam_size 1 \
        --text_maxlength 512 \
        --answer_maxlength 512 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint_final


