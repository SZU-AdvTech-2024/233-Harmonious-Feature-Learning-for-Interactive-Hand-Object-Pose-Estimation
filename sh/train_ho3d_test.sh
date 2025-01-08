        nohup\
        python traineval.py --HO3D_root /home/dell/Vscode/dataset/HO3D/data \
        --host_folder  /home/dell/Vscode/HFL-Net \
        --dex_ycb_root /home/dell/Vscode/dataset/DexYCB/data \
        --epochs 70 \
        --inp_res 256 \
        --lr 1e-4 \
        --train_batch 46 \
        --mano_lambda_regulshape 0 \
        --mano_lambda_regulpose  0 \
        --lr_decay_gamma 0.7 \
        --lr_decay_step 10 \
        --test_batch 46 \
        --use_ho3d \
        --evaluate \
        --resume /home/dell/Vscode/HFL-Net/46epochnewcheckpoint/checkpoint_70.pth.tar \
        > train_check_ho3d_test.log 2>&1 &

        #        CUDA_VISIBLE_DEVICES=0,1,3,4\
        # /home/dell/Vscode/HFL-Net/download/checkpoint_ho3d.pth.tar  /home/dell/Vscode/HFL-Net/download/checkpoint_ho3d.pth.tar
        
