@echo off
@REM python main.py config/config_cnn_2.yaml
@REM python main.py config/config_res_1.yaml
@REM python main.py config/config_vit_1.yaml

@REM python test.py config/config_vit_2.yaml
@REM python test.py config/config_vit_3.yaml
@REM python test.py config/config_vit_4.yaml
@REM python test.py config/config_vit_5.yaml
@REM python test.py config/config_vit_6.yaml
@REM python test.py config/config_vit_7.yaml
@REM python test.py config/config_vit_8.yaml
@REM python test.py config/config_vit_9.yaml
@REM python test.py config/config_vit_10.yaml
@REM python test.py config/config_vit_11.yaml

python main_pack.py config/config_pack_vit_1.yaml

@REM python main_ensemble.py config/config_ensemble_1.yaml
@REM python main_ensemble.py config/config_ensemble_3.yaml
pause