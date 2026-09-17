
#!/bin/bash

python predict.py \
    --dataset vocaset \
    --vertice_dim 15069 \
    --feature_dim 256 \
    --output_fps 30 \
    --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" \
    --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" \
    --model_name "face_diffuser1_vocaset_50" \
    --fps 30 \
    --condition "FaceTalk_170728_03272_TA" \
    --subject "FaceTalk_170731_00024_TA" \
    --diff_steps 500 \
    --gru_dim 256 \
    --wav_path "test.wav"