import matplotlib.pyplot as plt
import pyedflib
import sourcedata as sd


# read eeg signals from a sham mouse and a TBI mouse
sham_edf = pyedflib.EdfReader('data/edf/Sham102_BL5.edf')
tbi_edf = pyedflib.EdfReader('data/edf/TBI101_BL5.edf')
sham_eeg = sham_edf.readSignal(0)
tbi_eeg = tbi_edf.readSignal(0)
sham_edf._close()
tbi_edf._close()

# read stage data from the same sham and TBI mice
# assign: W -> 0, NR -> 1, R -> 2
sham_stage = open('data/sleep_staging/Sham102_BL5_Stages.csv')
tbi_stage = open('data/sleep_staging/TBI101_BL5_Stages.csv')
sham_stages = [line.split(',')[2] for line in sham_stage.readlines()[22:21623]]
tbi_stages = [line.split(',')[2] for line in tbi_stage.readlines()[22:21623]]
for i in range(len(sham_stages)):
    if sham_stages[i] == 'W':
        sham_stages[i] = 0
    elif sham_stages[i] == 'NR':
        sham_stages[i] = 1
    elif sham_stages[i] == 'R':
        sham_stages[i] = 2
for i in range(len(tbi_stages)):
    if tbi_stages[i] == 'W':
        tbi_stages[i] = 0
    elif tbi_stages[i] == 'NR':
        tbi_stages[i] = 1
    elif tbi_stages[i] == 'R':
        tbi_stages[i] = 2

# set up x-axis for plots
t = [x / 256 for x in range(len(sham_eeg))]
ts = [x * 4 for x in range(len(sham_stages))]

# set up plots
ax1 = plt.subplot(411)
plt.title('Sham102 and TBI101')
plt.plot(t, sham_eeg)
plt.ylabel('Sham EEG (uV)')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(412, sharex=ax1)
plt.plot(ts, sham_stages)
plt.ylabel('Sham Stage')
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(413, sharex=ax1)
plt.plot(t, tbi_eeg)
plt.ylabel('TBI EEG (uV)')
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = plt.subplot(414, sharex=ax1)
plt.plot(ts, tbi_stages)
plt.ylabel('TBI stage')
plt.xlabel('time (s)')
# plt.xlim(1200, 1300)
plt.show()
