save_path="../../results/cifar10_all-cnn-c_ckn_ulr-sgo_sgo_square/"
gpu=0
loss=square
num_iters=1000
seeds=(1 2 3 4 5 6 7 8 9 10)

num_filters=8
batch_size=4096
opt_method=ulr-sgo
step_size_method=fixed
step_size=-5

for seed in "${seeds[@]}"
do
    python cifar10_all-cnn-c_ckn.py --batch_size $batch_size --data_path ../../data/cifar10_whitened \
    --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=8
batch_size=4096
opt_method=sgo
step_size_method=fixed
step_size=-7

for seed in "${seeds[@]}"
do
    python cifar10_all-cnn-c_ckn.py --batch_size $batch_size --data_path ../../data/cifar10_whitened \
    --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=128
batch_size=256
opt_method=ulr-sgo
step_size_method=fixed
step_size=-6

for seed in "${seeds[@]}"
do
    python cifar10_all-cnn-c_ckn.py --batch_size $batch_size --data_path ../../data/cifar10_whitened \
    --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=128
batch_size=256
opt_method=sgo
step_size_method=fixed
step_size=-6

for seed in "${seeds[@]}"
do
    python cifar10_all-cnn-c_ckn.py --batch_size $batch_size --data_path ../../data/cifar10_whitened \
    --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done
