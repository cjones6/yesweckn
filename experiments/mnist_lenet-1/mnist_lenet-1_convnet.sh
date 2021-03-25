save_path="../../results/mnist_lenet-1_convnet_final/"
gpu=0,2
num_iters=10000
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=16384
hessian_reg=-4
lambda_filters=-7
step_size=-3
step_size_method=best-convnet-8

for seed in "${seeds[@]}"
do
    python mnist_lenet-1_convnet.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --lambda_filters $lambda_filters --num_filters $num_filters --num_iters $num_iters \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=16
batch_size=8192
hessian_reg=-6
lambda_filters=-6
step_size=-4
step_size_method=best-convnet-16

for seed in "${seeds[@]}"
do
    python mnist_lenet-1_convnet.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --lambda_filters $lambda_filters --num_filters $num_filters --num_iters $num_iters \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=32
batch_size=4096
hessian_reg=-2
lambda_filters=-8
step_size=-3
step_size_method=best-convnet-32

for seed in "${seeds[@]}"
do
    python mnist_lenet-1_convnet.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --lambda_filters $lambda_filters --num_filters $num_filters --num_iters $num_iters \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=64
batch_size=2048
hessian_reg=-3
lambda_filters=-7
step_size=-3
step_size_method=best-convnet-64

for seed in "${seeds[@]}"
do
    python mnist_lenet-1_convnet.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --lambda_filters $lambda_filters --num_filters $num_filters --num_iters $num_iters \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done


num_filters=128
batch_size=1024
hessian_reg=-2
lambda_filters=-8
step_size=-2
step_size_method=best-convnet-128

for seed in "${seeds[@]}"
do
    python mnist_lenet-1_convnet.py --batch_size $batch_size --data_path ../../data/mnist --gpu $gpu \
    --hessian_reg $hessian_reg --lambda_filters $lambda_filters --num_filters $num_filters --num_iters $num_iters \
    --save_path $save_path --seed $seed --step_size $step_size --step_size_method $step_size_method
done
