CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_wheat_imbal --phase train --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_maize_imbal --phase train --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_rice_imbal --phase train --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_sorg_imbal --phase train --data_path /opt/data1/dyw/GrainSet/sorg
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_wheat_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_maize_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_rice_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name vit_base --save_name vit_base_sorg_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/sorg