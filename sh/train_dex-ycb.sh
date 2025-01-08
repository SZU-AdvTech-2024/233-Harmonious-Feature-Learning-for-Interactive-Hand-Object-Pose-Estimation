        nohup\
        python traineval.py --HO3D_root /home/dell/Vscode/dataset/HO3D/data \
        --host_folder  /home/dell/Vscode/HFL-Net \
        --dex_ycb_root /home/dell/Vscode/dataset/DexYCB/data \
        --epochs 40 \
        --inp_res 256 \
        --lr 1e-4 \
        --train_batch 46 \
        --mano_lambda_regulshape 0 \
        --mano_lambda_regulpose  0 \
        --lr_decay_gamma 0.7 \
        --lr_decay_step 5 \
        --test_batch 46 \
        --use_cuda 1 \
        > train_check.log 2>&1 &
                #--CUDA_VISIBLE_DEVICES=0\
                
        
