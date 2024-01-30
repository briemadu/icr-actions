#!/bin/bash

######################## Modeling what to ask in iCRs #########################

# base clip overhearer 
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_base -only_icr_turns 

# clip overhearer random baseline
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_random -random_baseline -no_instruction -only_icr_turns 

# clip overhearer with scene before
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-scene_before -use_scene_before -only_icr_turns 

# clip overhearer with scene after
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-scene_after -use_scene_after -only_icr_turns 

# clip overhearer with scene before and scene after
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-both_scenes -use_scene_before -use_scene_after -only_icr_turns 

# clip overhearer with teacher-forced actions
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_teacher -actions_for_icr gold -only_icr_turns

# clip overhearer with teacher-forced actions and scene before
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_teacher_scene-before -actions_for_icr gold -use_scene_before -only_icr_turns 

# clip overhearer with all inputs
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_all-inputs -actions_for_icr gold -use_scene_before -use_scene_after -only_icr_turns 

# clip overhearer ablation with scenes but no instruction
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_ablation -no_instruction -use_scene_before -use_scene_after -only_icr_turns 

# clip overhearer context 0 to context 5
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-context0 -context_size 0 -only_icr_turns 
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-context1 -context_size 1 -only_icr_turns 
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-context2 -context_size 2 -only_icr_turns 
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-context3 -context_size 3 -only_icr_turns 
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-context4 -context_size 4 -only_icr_turns 
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-context5 -context_size 5 -only_icr_turns 

# clip overhearer long run
#python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_long -n_epochs 50 -only_icr_turns 

################################# Testing H3 #################################

# clip iCR-Action-Makers
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene -only_icr_turns 
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_logits -actions_for_icr logits -only_icr_turns 
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_teacher -actions_for_icr gold -only_icr_turns

python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_base -use_scene_before -only_icr_turns 
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_logits -use_scene_before -actions_for_icr logits -only_icr_turns 
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_teacher -use_scene_before -actions_for_icr gold -only_icr_turns 

# clip iCR-Action-Detecters
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_base -use_scene_before -use_scene_after -only_icr_turns 
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_logits -use_scene_before -use_scene_after -actions_for_icr logits -only_icr_turns 
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_teacher -use_scene_before -use_scene_after -actions_for_icr gold -only_icr_turns 


# TODO: run overhearer with checkpoint from action maker

python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_base_full -full_trf_encoder
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_teacher_full -actions_for_icr gold -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_no-scene_full -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_no-scene_logits_full -actions_for_icr logits -full_trf_encoder
python3 main.py -predict_icrs_turn -comet_tag turn_icr-action-maker_no-scene_teacher_full -actions_for_icr gold -full_trf_encoder
python3 main.py -predict_icrs_turn -dont_make_actions -comet_tag turn_overhearer_random_full -random_baseline -no_instruction -full_trf_encoder
python3 main.py -comet_tag turn_action-maker_random_full -random_baseline -no_instruction -full_trf_encoder

python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_base_full -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_random_full -random_baseline -no_instruction -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-scene_before_full -use_scene_before -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-scene_after_full -use_scene_after -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_teacher_full -actions_for_icr gold -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer-both_scenes_full -use_scene_before -use_scene_after -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_teacher_scene-before_full -actions_for_icr gold -use_scene_before -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_all-inputs_full -actions_for_icr gold -use_scene_before -use_scene_after -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -dont_make_actions -comet_tag clip_overhearer_ablation_full -no_instruction -use_scene_before -use_scene_after -only_icr_turns -full_trf_encoder

python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_logits_full -actions_for_icr logits -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_teacher_full -actions_for_icr gold -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_no-scene_full -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_base_full -use_scene_before -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_logits_full -use_scene_before -actions_for_icr logits -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-maker_teacher_full -use_scene_before -actions_for_icr gold -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_base_full -use_scene_before -use_scene_after -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_logits_full -use_scene_before -use_scene_after -actions_for_icr logits -only_icr_turns -full_trf_encoder
python3 main.py -predict_icrs_clipart -comet_tag clip_icr-action-detector_teacher_full -use_scene_before -use_scene_after -actions_for_icr gold -only_icr_turns -full_trf_encoder