import matplotlib.pyplot as plt
import preprocess as pp
import pyedflib

edf_filename = 'data/edf/Sham102_BL5.edf'
stage_filename = 'data/sleep_staging/Sham102_BL5_Stages.csv'

edf_file = pyedflib.EdfReader(edf_filename)
eeg_signal = edf_file.readSignal(0)
edf_file._close()
bpf_eeg_signal = pp.butter_bandpass_filter(eeg_signal, 0.5, 40, 256)

t = [x / 1024 for x in range(len(eeg_signal))]
ax1 = plt.subplot(411)
plt.plot(t, eeg_signal)
plt.ylabel('EEG signal (uV)')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(412, sharex=ax1)
plt.plot(t, bpf_eeg_signal)
plt.ylabel('BPF EEG signal (uV)')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlim(0, 2)
plt.show()
