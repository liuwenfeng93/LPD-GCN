for i in $(seq 1 9)
do 
python main.py --dataset $1 --fold_idx $i --iters_per_epoch $2 --epochs 300 --batch_size 64 --filename $1$i.result
done

