save_path="../../results/cifar10_all-cnn-c_ckn_svd_newton_final/"
gpu=0,2
num_iters=10000
nums_newton_iters=(0 50)
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=4096
hessian_reg=-6
step_size=-9
step_size_method=fixed

for num_newton_iters in "${nums_newton_iters[@]}"
do
    for seed in "${seeds[@]}"
    do
        python cifar10_all-cnn-c_ckn.py --batch_size $batch_size --data_path ../../data/cifar10_whitened --gpu $gpu \
        --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters \
        --num_newton_iters $num_newton_iters --save_path $save_path --seed $seed --step_size $step_size \
        --step_size_method $step_size_method
    done
done


num_filters=128
batch_size=256
hessian_reg=-5
step_size=-6
step_size_method=fixed

for num_newton_iters in "${nums_newton_iters[@]}"
do
    for seed in "${seeds[@]}"
    do
        python cifar10_all-cnn-c_ckn.py --batch_size $batch_size --data_path ../../data/cifar10_whitened --gpu $gpu \
        --hessian_reg $hessian_reg --num_filters $num_filters --num_iters $num_iters \
        --num_newton_iters $num_newton_iters --save_path $save_path --seed $seed --step_size $step_size \
        --step_size_method $step_size_method
    done
done
