import os
import os.path
import sys
import scipy.io.wavfile
import scipy
import numpy as np
import sh
import cPickle
import scipy.signal

instruments = [
#    'piano',          # 001
#    'marimba',        # 013
#    'organ',          # 018
#    'nylon_guitar',   # 025
#    'dist_guitar',    # 031
#    'aco_bass',       # 033
#    'syn_bass',       # 039
#    'strings',        # 052
#    'choir',          # 053
#    'brass',          # 063
#    'flute',          # 073
#    'synth',          # 082
#    'crystal',        # 099
#    'steel_drum',     # 115
#    'scream',         # SFX/098
#    '808'             # Drums/026
    ]

instruments = ['syn_bass', 'marimba', 'synth']

#pitches = range(24, 107)
#pitches = ([36, 39, 40, 44, 46], range(36, 48), range(55, 72), range(72, 91))
pitches = (range(36, 48), range(55, 72), range(72, 91))
pitch_map = [{name: index for index, name in enumerate(pitches_per_instr)} for pitches_per_instr in pitches]
#velocities = [60, 100, 127]
#velocities = [60, 127]
velocities = [127]

instr_map = {name: index for index, name in enumerate(instruments)}
vel_map = {velocity: index for index, velocity in enumerate(velocities)}

pitch_index_offsets = np.cumsum([0] + map(len, pitches))

if __name__ == '__main__':
    root = sys.argv[1]

    samples = [None] * (pitch_index_offsets[-1] * len(velocities))

    filenames = [''] * (pitch_index_offsets[-1] * len(velocities))

    for path in sh.find(root, '-name', '*.wav'):
        path = path.strip()
        parts = path.split('/')
        try:
            instrument = parts[-2]
            instr_index = instr_map[instrument]
            note = parts[-1].split('.')[0]
            pitch, velocity = map(int, note.split('_'))
            vel_index = vel_map[velocity]
            pitch_index = pitch_map[instr_index][pitch]

        except Exception:
            continue

        index = pitch_index_offsets[instr_index] * len(velocities) + \
            pitch_index * len(velocities) + vel_index
        sr, audio = scipy.io.wavfile.read(path)
        samples[index] = audio
        filenames[index] = path

    for i, path in enumerate(filenames):
        print '%d: %s' % (i, path)

    cPickle.dump(samples, open('samples.pkl', 'w'))
