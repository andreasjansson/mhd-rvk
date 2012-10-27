import math
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
import scipy
import numpy as np
import scipy.signal as signal
import cPickle
import onsetdetection
import scipy.io.wavfile
import os.path
import pdb
import random
import sys
from copy import copy
import operator
import sys, os
import keydetection

import dump_samples

samples = cPickle.load(open('samples.pkl', 'r'))

def get_target(audio, onsets, sr):
    onsets = onsets * sr
    lengths = onsets[1:] - onsets[:-1]
    np.append(lengths, len(audio) - onsets[-1])
    # make them powers of 2 for faster fft
    lengths = np.power(2, np.floor(np.log2(lengths)))
    spectra = []
    for onset, length in zip(onsets, lengths):
        if length > sr:
            length = sr
        spectra.append(get_spectrum(audio[int(onset) : int(onset + length)]))
    return spectra

def get_spectrum(audio):
    #audio = audio * np.hanning(len(audio))
    spectrum = abs(scipy.fft(audio))
    spectrum = np.array(spectrum[0:len(spectrum) / 2])
    return spectrum

def spectrum_similarity(spec1, spec2):
    spec1 = spec1 / np.max(spec1)
    spec2 = spec2 / np.max(spec2)
    diff = abs(spec1 - spec2)
    similarity = np.sum(max(diff) - diff)
    return similarity

def downsample(sig, factor):
    '''
    Low-pass filter using simple FIR, then pick every n sample, where n is
    the downsampling factor.
    '''
    fir = signal.firwin(61, 1.0 / factor, window="hamming")
    sig2 = np.convolve(sig, fir, mode="valid")
    sig2 = [int(x) for i, x in enumerate(sig2) if i % factor == 0]
    return sig2

def audio_from_chromosome(chromosome, length):
    audio = np.array([0] * length)
    if sum(chromosome) == 0:
        return audio

    for i, cell in enumerate(chromosome):
        if cell:
            sample = np.array(samples[i][0:length])
            audio += sample
    return (audio / float(max(audio))) * (2 ** 15)

class Evaluator:

    def __init__(self, target_spectrum):
        self.target_spectrum = target_spectrum
        self.target_nklang = keydetection.Chromagram.from_spectrum(filter_spectrum(target_spectrum), 11025).get_nklang()

    def evaluate(self, chromosome):

        if sum(chromosome.genomeList) == 0:
            return 0

        split_indices = np.cumsum([0] + map(len, dump_samples.pitches[:-1]))
        parts = np.split(chromosome.genomeList, split_indices)
        max_notes_per_part = [3, 1, 3, 3]
        for part, max_notes in zip(parts, max_notes_per_part):
            if sum(part) > max_notes:
                return 0

        audio = audio_from_chromosome(chromosome, len(self.target_spectrum) * 2)
        spectrum = get_spectrum(audio)

        filtered_spectrum = filter_spectrum(spectrum)
        nklang = keydetection.Chromagram.from_spectrum(filtered_spectrum, 11025).get_nklang()
        
        similarity = nklang_similarity(nklang, self.target_nklang) * spectrum_similarity(spectrum, self.target_spectrum)

        return similarity


def filter_spectrum(spectrum):
    filtered_spectrum = keydetection.SpectrumQuantileFilter(99).filter(spectrum)
    filtered_spectrum = keydetection.SpectrumGrainFilter().filter(spectrum)
    return filtered_spectrum
    
def nklang_similarity(a, b):
    matches = np.sum(np.sort(a.notes) == np.sort(b.notes))
    return (matches ** 2) + .1


if __name__ == '__main__':

    if os.path.exists('target.pkl') and os.path.exists('onsets.pkl'):
        onsets = cPickle.load(open('onsets.pkl', 'r'))
        target = cPickle.load(open('target.pkl', 'r'))
    else:
        sr, audio = scipy.io.wavfile.read('fembots.wav')
        sr = float(sr)
        audio = audio[:,0]
        onsets = onsetdetection.detect_onsets(audio)
        onsets = np.array(map(float, onsets))
        onsets /= sr

        audio = downsample(audio, 4)
        sr /= 4
        target = get_target(audio, onsets, sr)
        cPickle.dump(target, open('target.pkl', 'w'))
        cPickle.dump(onsets, open('onsets.pkl', 'w'))

    target_index = int(sys.argv[1])
    evaluator = Evaluator(target[target_index])

    samples = cPickle.load(open('samples.pkl', 'r'))

    def initialisator(genome, **args):
       genome.clearList()

       for i in xrange(len(samples)):
          choice = int(round(random.random() * .7))
          genome.append(choice)

    genome = G1DBinaryString.G1DBinaryString(len(samples))
    genome.evaluator.set(evaluator.evaluate)
    genome.initializator.set(initialisator)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setGenerations(100)
    ga.evolve(freq_stats=10)
    chromosome = ga.bestIndividual()
    audio = audio_from_chromosome(chromosome, (onsets[target_index + 1] - onsets[target_index]) * 11025)
    scipy.io.wavfile.write('audio_%d.wav' % target_index, 11025, np.array(audio, dtype = np.int16) / 2)
    cPickle.dump(chromosome.genomeList, open('chromosome.pkl', 'w'))



def play_audio(audio, sr):
    filename = 'tmp_play.wav'
    audio = np.array((audio / float(np.max(audio))) * (2 ** 15), dtype = np.int16)
    print np.max(audio)
    scipy.io.wavfile.write(filename, sr, audio)
    import sh
    sh.play(filename)
