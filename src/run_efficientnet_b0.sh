CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_wheat_imbal --phase train --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_maize_imbal --phase train --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_rice_imbal --phase train --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_sorg_imbal --phase train --data_path /opt/data1/dyw/GrainSet/sorg
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_wheat_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_maize_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_rice_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name efficientnet_b0 --save_name efficientnet_b0_sorg_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/sorg
