save_path="../../results/mnist_lenet-5_convnet_final_incomplete_complete/"
gpu=0,2
num_iters=10000
seeds=(1 2 3 4 5 6 7 8 9 10)


batch_size=16384
hessian_reg=-5
incomplete_schemes=(0 1)
lambda_filters=-7
step_size=-2
step_size_method=best-convnet-incomplete

for incomplete_scheme in "${incomplete_schemes[@]}"
do
    for seed in "${seeds[@]}"
    do
        python mnist_lenet-5_convnet.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
        --hessian_reg $hessian_reg --incomplete_scheme $incomplete_scheme --lambda_filters $lambda_filters \
        --num_iters $num_iters --save_path $save_path --seed $seed --step_size $step_size \
        --step_size_method $step_size_method
    done
done
