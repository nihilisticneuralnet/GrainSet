CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_wheat_imbal --phase train --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_maize_imbal --phase train --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_rice_imbal --phase train --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_sorg_imbal --phase train --data_path /opt/data1/dyw/GrainSet/sorg
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_wheat_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/wheat
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_maize_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/maize
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_rice_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/rice
CUDA_VISIBLE_DEVICES=4,7 python src/train.py --model_name inception_v3 --save_name inception_v3_sorg_bal --phase train_bal --data_path /opt/data1/dyw/GrainSet/sorg