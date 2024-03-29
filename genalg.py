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
import threading
import dump_samples
from copy import copy

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

def audio_from_chromosome(chromosome, length, offset = 0):
    audio = np.array([0] * length)
    if sum(chromosome) == 0:
        return audio

    for i, cell in enumerate(chromosome):
        if cell:
            sample = np.array(samples[i + offset][0:length])
            audio += sample
    return (audio / float(max(audio))) * (2 ** 15)

def chroma_over_pitches(spectrum, pitches, sr):
    spectrum = copy(spectrum)
    midi0 = 16.3515978312
    low = midi0 * math.pow(2.0, pitches[0] / 12.0)
    high = midi0 * math.pow(2.0, pitches[-1] / 12.0)
    low_bin = int(math.floor((low / sr) * len(spectrum) * 2))
    high_bin = int(math.ceil((high / sr) * len(spectrum) * 2))
    for i in range(0, low_bin) + range(high_bin, len(spectrum)):
        spectrum[i] = 0
    return keydetection.Chromagram.from_spectrum(spectrum, 11025)

class Evaluator:

    def __init__(self, target_spectrum):
        self.target_spectrum = target_spectrum
        filtered_spectrum = filter_spectrum(target_spectrum)
        self.target_chromagrams = [chroma_over_pitches(filtered_spectrum, pitches, 11025)
                                   for pitches in dump_samples.pitches]

    def evaluate(self, chromosome):

        if sum(chromosome.genomeList) == 0:
            return 0

        split_indices = np.cumsum([0] + map(len, dump_samples.pitches[:-1]))
        parts = np.split(chromosome.genomeList, split_indices)[1:]
        max_notes_per_part = [1, 3]
        for part, max_notes in zip(parts, max_notes_per_part):
            if sum(part) > max_notes:
                return 0

        total_similarity = 0
        for i, part in enumerate(parts):
            audio = audio_from_chromosome(part, len(self.target_spectrum) * 2, split_indices[i])
            spectrum = get_spectrum(audio)
            chromagram = chroma_over_pitches(spectrum, dump_samples.pitches[i], 11025)
            similarity = chromagram_similarity(self.target_chromagrams[i], chromagram)
            total_similarity += similarity

        return total_similarity


def filter_spectrum(spectrum):
    filtered_spectrum = keydetection.SpectrumQuantileFilter(99).filter(spectrum)
    filtered_spectrum = keydetection.SpectrumGrainFilter().filter(spectrum)
    return filtered_spectrum
    
def nklang_similarity(a, b):
    matches = np.sum(np.sort(a.notes) == np.sort(b.notes))
    return (matches ** 2) + .1

def chromagram_similarity(a, b):
    a = a.values / max(a.values)
    b = b.values / max(b.values)
    return np.sum(np.power(np.abs(np.subtract(a, b)), 2))

class GenAlgThread(threading.Thread):

    def __init__(self, target_index, onsets, target):
        threading.Thread.__init__(self)

        self.target_index = target_index
        self.evaluator = Evaluator(target[target_index])
        self.onsets = onsets
        self.target = target

    def run(self):

        def initialisator(genome, **args):
           genome.clearList()
           for i in xrange(len(samples)):
              choice = int(round(random.random() * .7))
              genome.append(choice)
        
        genome = G1DBinaryString.G1DBinaryString(len(samples))
        genome.evaluator.set(self.evaluator.evaluate)
        genome.initializator.set(initialisator)
        ga = GSimpleGA.GSimpleGA(genome)
        ga.setGenerations(100)
        ga.evolve(freq_stats=10)
        chromosome = ga.bestIndividual()
        audio = audio_from_chromosome(chromosome, (self.onsets[self.target_index + 1] - self.onsets[self.target_index]) * 11025)
        scipy.io.wavfile.write('audio_%d.wav' % self.target_index, 11025, np.array(audio, dtype = np.int16) / 2)
        cPickle.dump(chromosome.genomeList, open('chromosome.pkl', 'w'))


if __name__ == '__main__':

    if os.path.exists('target.pkl') and os.path.exists('onsets.pkl'):
        onsets = cPickle.load(open('onsets.pkl', 'r'))
        target = cPickle.load(open('target.pkl', 'r'))
    else:
        sr, audio = scipy.io.wavfile.read('dancing.wav')
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

    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])

    for i in range(start_index, end_index):
        GenAlgThread(i, onsets, target).start()

def play_audio(audio, sr):
    filename = 'tmp_play.wav'
    audio = np.array((audio / float(np.max(audio))) * (2 ** 15), dtype = np.int16)
    print np.max(audio)
    scipy.io.wavfile.write(filename, sr, audio)
    import sh
    sh.play(filename)
