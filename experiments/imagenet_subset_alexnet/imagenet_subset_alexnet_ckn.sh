save_path="../../results/imagenet_subset_alexnet_ckn_final/"
gpu=0,2
num_iters=10000
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=512
hessian_reg=-5
step_size=-6
step_size_method=best-ckn-8

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg \
    --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed --step_size $step_size \
    --step_size_method $step_size_method
done


num_filters=16
batch_size=512
hessian_reg=-4
step_size=-6
step_size_method=best-ckn-16

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg \
    --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed --step_size $step_size \
    --step_size_method $step_size_method
done


num_filters=32
batch_size=512
hessian_reg=-5
step_size=-6
step_size_method=best-ckn-32

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg \
    --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed --step_size $step_size \
    --step_size_method $step_size_method
done


num_filters=64
batch_size=256
hessian_reg=-5
step_size=-6
step_size_method=best-ckn-64

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg \
    --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed --step_size $step_size \
    --step_size_method $step_size_method
done


num_filters=128
batch_size=128
hessian_reg=-5
step_size=-6
step_size_method=best-ckn-128

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg \
    --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed --step_size $step_size \
    --step_size_method $step_size_method
done
