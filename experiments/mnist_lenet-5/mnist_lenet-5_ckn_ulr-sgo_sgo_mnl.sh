save_path="../../results/mnist_lenet-5_ckn_ulr-sgo_sgo_final_mnl/"
gpu=0
loss=cross-entropy
num_iters=1000
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=16384
hessian_reg=-7
step_size=-7
step_size_method=fixed
opt_method=ulr-sgo

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=8
batch_size=16384
hessian_reg=-7
step_size=-7
step_size_method=fixed
opt_method=sgo

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=128
batch_size=1024
hessian_reg=-6
step_size=-7
step_size_method=fixed
opt_method=ulr-sgo

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=128
batch_size=1024
hessian_reg=-6
step_size=-3
step_size_method=fixed
opt_method=sgo

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done
