#!/bin/bash

# pure Action-Maker, only at iCR turns
python3 main.py -comet_tag clip_action-maker_base -only_icr_turns 
python3 main.py -comet_tag clip_action-maker_scene-before -use_scene_before -only_icr_turns 

# pure Action-Detecter, only at iCR turns
python3 main.py -comet_tag clip_action-detecter -use_scene_before -use_scene_after -only_icr_turns 
