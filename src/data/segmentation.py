import matplotlib.pyplot as plt
import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave


"""
A script that iterates through the wav files and uses pyAudioAnalysis' silence extraction module to make wav files of segments when the participant is speaking and concatenates them into a single wave file.
"""

def remove_silence(filename, out_dir, smoothing=1.0, weight=0.3, plot=False):
    """
    A function that implements pyAudioAnalysis' silence extraction module and creates wav files of the non-silent portions of audio. The smoothing and weight parameters were tuned for the AVEC 2016 data set.

    Parameters
    ----------
    filename : string
        path to the input wav file
    out_dir : string
        path to the desired directory (where an audio_segments folder will be created)
    smoothing : float
        used for smoothing in the SVM #Ky - DIG INTO THE SOURCE CODE
    weight : float
        probability threshold for silence removal
    plot : bool
        plots SVM probabilities of silence (used in troubleshooting)

    Returns
    -------
    A folder for each particpant containting a single wave file (named 'PXXX_no_silence.wav') with the vast majority of silence and virtual interviewer speech removed. Feature exctraction can be performed on these files.
    """
    partic_id = 'P' + filename.split('/')[-1].split('_')[0] # PXXX
    # create participant directory for segmented wav files
    if is_segmentable(partic_id):
        participant_dir = os.path.join(out_dir, partic_id)
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)

        os.chdir(participant_dir) # change to dir to write segmented files

        [Fs, x] = aIO.readAudioFile(filename)
        segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow=smoothing, Weight=weight, plot=plot)

        for s in segments:
            seg_name = "{:s}_{:.2f}-{:.2f}.wav".format(partic_id, s[0], s[1])
            wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

        # concatentate segmented wave files within participant directory
        concatentate_segments(participant_dir, partic_id)

def is_segmentable(partic_id):
    """
    A function that returns True if the participant's interview clip is not in the manually identified set of troubled clips. The clips below were not segmentable do to excessive static, proximity to the virtual interviewer, volume levels, etc.
    """
    troubled = set(['P300', 'P305', 'P306', 'P308', 'P315', 'P316'])
    return partic_id not in troubled

def concatentate_segments(participant_dir, partic_id, remove_segment=True):
    """
    A fucntion that concatentates all the wave files in a participants directory in to single wav file and writes in to the participant's direction, then removes the individual segments (when remove_segment=True)
    """
    infiles = os.listdir(participant_dir) # list of wav files in directory
    outfile = '{}_no_silence.wav'.format(partic_id)

    data = []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
        if remove_segment:
            os.remove(infile)

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0]) # details of the files must be the same (channel, frame rates, etc.)

    # write each segment to output
    for idx in range(len(data)):
        output.writeframes(data[idx][1])
    output.close()

if __name__ == '__main__':
    dir_name = '/Users/ky/Desktop/depression-detect/data/raw/audio' # directory containing wav files
    out_dir = '/Users/ky/Desktop/depression-detect/data/interim'

    for file in os.listdir(dir_name):
        if file.endswith('.wav'):
            filename = os.path.join(dir_name, file)
            remove_silence(filename, out_dir)