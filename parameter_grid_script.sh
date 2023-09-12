for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.8
do
    for b in 0.2 0.4 0.6 0.8
    do
        python scripts/main_tf_icon.py  --ckpt ./ckpt/v2-1_512-ema-pruned.ckpt  --root ./inputs/same_domain  --domain 'same'  --dpm_steps 50  --dpm_order 2 --scale 2.5  --tau_a $a  --tau_b $b  --outdir "./outputs/parameters/$a$b"   --gpu cuda:0                                                      --seed 3407
    done
done