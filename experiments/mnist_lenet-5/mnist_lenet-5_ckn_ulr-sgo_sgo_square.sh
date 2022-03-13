save_path="../../results/mnist_lenet-5_ckn_ulr-sgo_sgo_square/"
gpu=0
loss=square
num_iters=1000
seeds=(1 2 3 4 5 6 7 8 9 10)


num_filters=8
batch_size=16384
step_size_method=fixed
step_size=-8
opt_method=ulr-sgo

for seed in "${seeds[@]}"
do
  python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist \
   --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters \
  --opt_method $opt_method --save_path $save_path --seed $seed --step_size $step_size \
  --step_size_method $step_size_method
done


num_filters=8
batch_size=16384
step_size_method=fixed
step_size=-7
opt_method=sgo

for seed in "${seeds[@]}"
do
  python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist \
   --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters \
  --opt_method $opt_method --save_path $save_path --seed $seed --step_size $step_size \
  --step_size_method $step_size_method
done


num_filters=128
batch_size=1024
step_size_method=fixed
step_size=-6
opt_method=ulr-sgo

for seed in "${seeds[@]}"
do
  python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist \
   --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters \
  --opt_method $opt_method --save_path $save_path --seed $seed --step_size $step_size \
  --step_size_method $step_size_method
done


num_filters=128
batch_size=1024
step_size_method=fixed
step_size=-6
opt_method=sgo

for seed in "${seeds[@]}"
do
  python mnist_lenet-5_ckn.py --batch_size $batch_size --data_path ../../data/mnist \
   --gpu $gpu --loss $loss --num_filters $num_filters --num_iters $num_iters \
  --opt_method $opt_method --save_path $save_path --seed $seed --step_size $step_size \
  --step_size_method $step_size_method
done
