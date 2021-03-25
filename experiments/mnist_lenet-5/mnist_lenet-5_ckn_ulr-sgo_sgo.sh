save_path="../../results/mnist_lenet-5_ckn_ulr-sgo_sgo_final/"
gpu=0,2
num_iters=10000
opt_methods=(ulr-sgo sgo)
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=16384
hessian_reg=-7
step_size=-7
step_size_method=best-ckn-8

for opt_method in "${opt_methods[@]}"
do
    for seed in "${seeds[@]}"
    do
        python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
        --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
        --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
    done
done


num_filters=128
batch_size=1024
hessian_reg=-6
step_size=-7
step_size_method=best-ckn-128

for opt_method in "${opt_methods[@]}"
do
    for seed in "${seeds[@]}"
    do
        python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
        --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters --opt_method $opt_method \
        --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
    done
done
