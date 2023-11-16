#!/usr/env/bin/python3

import os
from os.path import join
import numpy as np
import soundfile as sf
import glob
import argparse
import json
import pyroomacoustics as pra
SEED = 100
np.random.seed(SEED)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_reverberant_file(s, sr, params, generate_target=False):
    
    room_dims = np.array(params["room_dims"])
    t60 = params["t60"]
    center_mic_position = np.array(params["center_mic_position"])
    source_position = np.array(params["source_position"])
    assert (center_mic_position < room_dims).all(), "Center of microphone array is not in room"
    assert (source_position < room_dims).all(), "Source is not in room"
    mic_array_radius = params["mic_array_radius"]
    mic_numbers = params["mic_numbers"]

    distance_source = 1/np.sqrt(3)*np.linalg.norm(center_mic_position - source_position)
    mic_array_2d = pra.beamforming.circular_2D_array(center_mic_position[: -1], mic_numbers, phi0=0, radius=mic_array_radius) # Compute microphone array
    mic_array = np.pad(mic_array_2d, ((0, 1), (0, 0)), mode="constant", constant_values=center_mic_position[-1])

    ### Reverberant Room
    e_absorption, max_order = pra.inverse_sabine(t60, room_dims) #Compute absorption coeff
    reverberant_room = pra.ShoeBox(
        room_dims, fs=sr, materials=pra.Material(e_absorption), max_order=min(3, max_order), ray_tracing=True
    ) # Create room
    reverberant_room.set_ray_tracing()
    reverberant_room.add_microphone_array(mic_array) # Add microphone array
    reverberant_room.add_source(source_position, signal=s) # Add source
    reverberant_room.compute_rir() # Compute RIR
    reverberant_room.simulate() # Generate reverberant
    t60_real = np.mean(reverberant_room.measure_rt60()).squeeze()
    y_reverberant = np.squeeze(np.array(reverberant_room.mic_array.signals))

    if generate_target:
        #compute target
        e_absorption_target = 0.8
        target_room = pra.ShoeBox(
            room_dims, fs=sr, materials=pra.Material(e_absorption_target), max_order=3
        ) # Create room
        target_room.add_microphone_array(mic_array) # Add microphone array
        target_room.add_source(source_position, signal=s) #Add source
        target_room.compute_rir() # Compute RIR
        target_room.simulate() # Generate reverberant target
        t60_real_target = np.mean(target_room.measure_rt60()).squeeze()
        y_target = np.squeeze(np.array(target_room.mic_array.signals))
        # Pad to match reverberant tail
        noise_floor_snr = 50
        noise_floor_power = 1/s.shape[0]*np.sum(s**2)*np.power(10,-noise_floor_snr/10)
        noise_floor_signal = np.random.rand(*y_reverberant.shape) * np.sqrt(noise_floor_power)
        y_target = np.pad(y_target, ((0, 0), (0, y_reverberant.shape[-1] - y_target.shape[-1])) )
        y_target = y_target + noise_floor_signal

        return y_reverberant, y_target

    else:
        return y_reverberant, None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', type=str, help="Input mono file", default="/data/lemercier/databases/WSJ0/wsj0/si_et_05/440/440c0214.wav")
    parser.add_argument('--params_idx', type=int, help="Choice of parameterization from reverb_params.json")
    parser.add_argument('--generate_target', action="store_true")
    args = parser.parse_args()

    s, sr = sf.read(args.speech)
    with open("reverb_params.json", "r") as j:
        params = json.load(j)[str(args.params_idx)]

    print("Generating reverberated file...")
    for key, param in params.items():
        print(key + " : " + str(param))

    y_reverberant, y_target = generate_reverberant_file(s, sr, params, generate_target=args.generate_target)

    ensure_dir("stimuli")
    sf.write(join("stimuli", os.path.basename(args.speech[:-4]) + f"_reverb_{args.params_idx}_t60={params['t60']}.wav"), y_reverberant.T, sr)
    if args.generate_target:
        sf.write(join("stimuli", os.path.basename(args.speech[:-4]) + f"_target_{args.params_idx}_t60={params['t60']}.wav"), y_target.T, sr)