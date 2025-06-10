# seed
seed_list=(0)
# gpu
gpu_list=(0) # Assign GPU numbers for each seed

#For var3, even though it is named super_resolution, it just trains overfitting image. 
#super_resolution will be held during evaluation. Also, you don't need to train overfitting.

# task num ['super_resolution', 'denoising', 'overfitting']
task_list=(1) 
# noise num [0.05 0.1]
noise_list=(0 1) # applies only when task num is 1

# Run python for n seeds and task variations
for seed_idx in "${!seed_list[@]}"
do
    seed="${seed_list[$seed_idx]}"
    gpu="${gpu_list[$seed_idx]}"

    for task_idx in "${!task_list[@]}"
    do
        task="${task_list[$task_idx]}"
        if [ "$task" -eq 1 ]; then
            # When task num is 1, iterate over noise num
            for noise_idx in "${!noise_list[@]}"
            do
                noise="${noise_list[$noise_idx]}"
                python run_DIV2K.py "$seed" "$gpu" "$task" "$noise"
                wait  # Ensure process completes
                echo "Executed with seed=$seed, gpu=$gpu, task=$task, noise=$noise"
            done
        else
            # When task num is 0, no noise num
            python run_DIV2K.py "$seed" "$gpu" "$task" "0"
            wait  # Ensure process completes
            echo "Executed with seed=$seed, gpu=$gpu, task=$task, noise=0"
        fi
    done

    # Clean up lingering Python processes
    pkill -9 -f run_DIV2K.py
done
