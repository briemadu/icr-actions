#!/bin/bash

######################### Modeling when to ask iCRs ###########################

# base turn overhearer 
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_base

# turn overhearer random baseline
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_random -random_baseline -no_instruction 

# turn overhearer with scene before
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-scene_before -use_scene_before
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-scene_before_full -use_scene_before -full_trf_encoder
# turn overhearer with scene after
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-scene_after -use_scene_after
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-scene_after_full -use_scene_after -full_trf_encoder
# turn overhearer with scene before and scene after
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-both_scenes -use_scene_before -use_scene_after
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-both_scenes_full -use_scene_before -use_scene_after -full_trf_encoder

# turn overhearer with teacher-forced actions
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_teacher -actions_for_icr gold
# turn overhearer with teacher-forced actions and scene before
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_teacher_scene-before -actions_for_icr gold -use_scene_before
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_teacher_scene-before_full -actions_for_icr gold -use_scene_before -full_trf_encoder
# turn overhearer with all inputs
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_all-inputs -actions_for_icr gold -use_scene_before -use_scene_after
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_all-inputs_full -actions_for_icr gold -use_scene_before -use_scene_after -full_trf_encoder

# turn overhearer ablation with scenes but no instruction
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_ablation -no_instruction -use_scene_before -use_scene_after
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_ablation_full -no_instruction -use_scene_before -use_scene_after -full_trf_encoder

# turn overhearer context 0 to context 5
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-context0 -context_size 0
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-context1 -context_size 1
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-context2 -context_size 2 
#python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-context3 -context_size 3 
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-context4 -context_size 4 
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer-context5 -context_size 5 

# turn overhearer long run
#python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_long -n_epochs 50


################################# Testing H1 #################################

# turn iCR-Action-Makers
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_no-scene
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_no-scene_logits -actions_for_icr logits
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_no-scene_teacher -actions_for_icr gold

python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_base -use_scene_before 
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_logits -use_scene_before -actions_for_icr logits
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_teacher -use_scene_before -actions_for_icr gold

python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_base_full -use_scene_before -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_logits_full -use_scene_before -actions_for_icr logits -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_teacher_full -use_scene_before -actions_for_icr gold -full_trf_encoder

# turn iCR-Action-Detecters
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-detector_base -use_scene_before -use_scene_after 
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-detector_logits -use_scene_before -use_scene_after -actions_for_icr logits
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-detector_teacher -use_scene_before -use_scene_after -actions_for_icr gold

python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-detector_base_full -use_scene_before -use_scene_after -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-detector_logits_full -use_scene_before -use_scene_after -actions_for_icr logits -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-detector_teacher_full -use_scene_before -use_scene_after -actions_for_icr gold -full_trf_encoder


################################# Testing H2 #################################

# random baseline
python3 main.py -comet_tag turn_action-maker_random -random_baseline -no_instruction 

# pure Action-Maker
python3 main.py -comet_tag turn_action-maker_base
python3 main.py -comet_tag turn_action-maker_scene-before -use_scene_before
python3 main.py -comet_tag turn_action-maker_scene-before_full -use_scene_before -full_trf_encoder

# pure Action-Detecter
python3 main.py -comet_tag turn_action-detecter -use_scene_before -use_scene_after
python3 main.py -comet_tag turn_action-detecter_full -use_scene_before -use_scene_after -full_trf_encoder