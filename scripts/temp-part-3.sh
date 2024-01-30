#!/bin/bash

######################## Fine-tuning from checkpoints #########################

# clip iCR-Action-Makers
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_finetuned -only_icr_turns -checkpoint ./comet-logs/cr-codraw-eacl24-manuscript/72add80aef93400ca790d79f044cdca3/checkpoints/val_action_BinaryAveragePrecision_epoch=7.ckpt
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_logits_finetuned -actions_for_icr logits -only_icr_turns -checkpoint ./comet-logs/cr-codraw-eacl24-manuscript/72add80aef93400ca790d79f044cdca3/checkpoints/val_action_BinaryAveragePrecision_epoch=7.ckpt

python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_base_finetuned -use_scene_before -only_icr_turns -checkpoint ./comet-logs/cr-codraw-eacl24-manuscript/e429dcaa12f34c7e9a2cb7646840e11e/checkpoints/val_action_BinaryAveragePrecision_epoch=11.ckpt
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_logits_finetuned -use_scene_before -actions_for_icr logits -only_icr_turns -checkpoint ./comet-logs/cr-codraw-eacl24-manuscript/e429dcaa12f34c7e9a2cb7646840e11e/checkpoints/val_action_BinaryAveragePrecision_epoch=11.ckpt

# clip iCR-Action-Detecters
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_base_finetuned -use_scene_before -use_scene_after -only_icr_turns -checkpoint ./comet-logs/cr-codraw-eacl24-manuscript/e880943167fa478fafe0538c3335cae6/checkpoints/val_action_BinaryAveragePrecision_epoch=11.ckpt
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_logits_finetuned -use_scene_before -use_scene_after -actions_for_icr logits -only_icr_turns -checkpoint ./comet-logs/cr-codraw-eacl24-manuscript/e880943167fa478fafe0538c3335cae6/checkpoints/val_action_BinaryAveragePrecision_epoch=11.ckpt
