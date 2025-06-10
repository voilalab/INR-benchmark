seed_list=(0 1 2 3 4)
gpu_list=(0 0 0 0 0)
# 0: Spheres, 1:Bandlimited, 2:Sierpinski, 3:StarTarget, 4:DragonFitting(VoxelFitting)
signal_list=(0 0 0 0 0)
# 2 or 3
dimension_list=(3 3 3 3 3)
# dummy variable except Dragon Fitting
sparse_list=(0 0 0 0 0)

n=${#seed_list[@]}

for i in $(seq 1 $n)
do
    var1="${seed_list[$i-1]}"  
    var2="${gpu_list[$i-1]}"  
    var3="${signal_list[$i-1]}"  
    var4="${dimension_list[$i-1]}"  
    var5="${sparse_list[$i-1]}"  

    # Run python with selected variables
    python run.py "$var1" "$var2" "$var3" "$var4" "$var5"

    echo "Iteration $i: Executed Python script with var1=$var1, var2=$var2, var3=$var3, var4=$var4, var5=$var5"
done
