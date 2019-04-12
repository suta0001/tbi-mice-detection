import extractfeatures as ef
import matplotlib.pyplot as plt
import pyedflib
import sourcedata as sd

edf_filename = 'data/edf/Sham102_BL5.edf'
stage_filename = 'data/sleep_staging/Sham102_BL5_Stages.csv'

edf_file = pyedflib.EdfReader(edf_filename)
eeg_signal = edf_file.readSignal(0)
edf_file._close()
eeg_epochs, stage_epochs = sd.create_epochs(4, edf_filename, stage_filename)
features = ef.calc_permutation_entropy(eeg_epochs, [3, 4, 5, 6, 7],
                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

t = [x / 1024 for x in range(len(eeg_signal))]
x = range(len(stage_epochs))
y0 = [features[i][0] for i in range(len(features))]
y1 = [features[i][49] for i in range(len(features))]
ax1 = plt.subplot(411)
plt.plot(t, eeg_signal)
plt.ylabel('EEG signal (uV)')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(412, sharex=ax1)
plt.plot(x, y0)
plt.ylabel('H(3, 1)')
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(413, sharex=ax1)
plt.plot(x, y1)
plt.ylabel('H(7, 10)')
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = plt.subplot(414, sharex=ax1)
plt.plot(x, stage_epochs)
plt.ylabel('sleep stage')
plt.xlabel('time epoch -> 4 s/epoch')
plt.xlim(1200, 1300)
plt.show()
