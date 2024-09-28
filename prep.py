import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

#waveform
signal, sr = librosa.load(file, sr = 22050) #numpy array having values equal to sr * T --> 22050 * 30 sec 
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#fft --> spectrum

fft = np.fft.fft(signal)

magnitude = np.abs(fft) 
#indicate the contribution of each frequency bins to the overalll sound
frequency = np.linspace(0, sr, len(magnitude))
#two arrays tells you how much frequency is contributing to the sound

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

# plt.plot(left_frequency, left_magnitude)

# plt.xlabel("Frequency")
# plt.ylabel("Magitude")
# plt.show()

#stft --> spectogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, n_fft = n_fft, hop_length = hop_length)



spectrogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectogram, sr =sr, hop_length = hop_length)


# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCCs

mfccs = librosa.feature.mfcc(y=signal, n_fft=n_fft, sr=sr, n_mfcc=40)

librosa.display.specshow(mfccs,  hop_length=hop_length, x_axis='time')
plt.colorbar()
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficients")
plt.show()








