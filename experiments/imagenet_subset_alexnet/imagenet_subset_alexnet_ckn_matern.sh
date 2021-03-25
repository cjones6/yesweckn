save_path="../../results/imagenet_subset_alexnet_ckn_matern_final/"
gpu=0,2
kernel=matern_sphere
num_iters=10000
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=512
bw=0.6
hessian_reg=-5
matern_order=2.5
step_size=-6
step_size_method=best-ckn-matern-8

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size --bw $bw \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg --kernel $kernel \
    --matern_order $matern_order --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed \
    --step_size $step_size --step_size_method $step_size_method
done


num_filters=16
batch_size=512
bw=0.7
hessian_reg=-5
matern_order=1.5
step_size=-9
step_size_method=best-ckn-matern-16

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size --bw $bw \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg --kernel $kernel \
    --matern_order $matern_order --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed \
    --step_size $step_size --step_size_method $step_size_method
done


num_filters=32
batch_size=512
bw=0.7
hessian_reg=-5
matern_order=1.5
step_size=-6
step_size_method=best-ckn-matern-32

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size --bw $bw \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg --kernel $kernel \
    --matern_order $matern_order --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed \
    --step_size $step_size --step_size_method $step_size_method
done


num_filters=64
batch_size=256
bw=0.9
hessian_reg=-6
matern_order=3.5
step_size=-7
step_size_method=best-ckn-matern-64

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size --bw $bw \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg --kernel $kernel \
    --matern_order $matern_order --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed \
    --step_size $step_size --step_size_method $step_size_method
done


num_filters=128
batch_size=128
bw=0.8
hessian_reg=-4
matern_order=2.5
step_size=-6
step_size_method=best-ckn-matern-128

for seed in "${seeds[@]}"
do
    python imagenet_subset_alexnet_ckn.py --batch_size $batch_size --bw $bw \
    --data_path /mnt/ssd/cjones6/ilsvrc2012_subset_subset --gpu $gpu --hessian_reg $hessian_reg --kernel $kernel \
    --matern_order $matern_order --num_filters $num_filters --num_iters $num_iters --save_path $save_path --seed $seed \
    --step_size $step_size --step_size_method $step_size_method
done
