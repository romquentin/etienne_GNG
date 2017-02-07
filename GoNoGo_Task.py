## GoNoGo task ##

import numpy as np
import os.path as op
import mne
from mne import Epochs, find_events
from mne.io import read_raw_ctf
import matplotlib.pyplot as plt
from path import path
from mne.epochs import concatenate_epochs

# Find the folder with MEG data
data_path = '/Users/Sallardef/Downloads/MEG-DATA/'
subject = 'Sub1_734'
task = 'Task'
##run = 'TCBTDAUP_task-gng_20160928_02.ds'

##fname_raw = op.join(data_path, subject, run)
##print(fname_raw)

# Open file
path_raw = op.join(data_path, subject, task)
runs = path(path_raw).glob('*.ds')

# Make sure there are 10 blocks. If not, Runtime Error will be written
if len(runs) != 10:
    raise RuntimeError('Could not find 10 runs in %s' % path_raw)

# Give the number of trigger and epoch - ???????
events_meg = list()
epochs_list = list()

# Loop for blocks
# for run_number, this_run in enumerate(runs):
run_number = 0
this_run = runs[0]
fname_raw = op.join(path_raw, this_run)
raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
events_meg_ = mne.find_events(raw)

# Pre-processing + epoch duration
raw.filter(l_freq=.1, h_freq=50.0, l_trans_bandwidth=.01)
tmin = -.100
tmax = 1.9

# Display the name of the trigger with the time
events = find_events(raw, stim_channel='UPPT001')
print (events)

# Trigger to use # (Go=1, NoGo=32)
event_id = dict (all=1)
# reject = dict(mag=4e-12)

epochs = Epochs(raw, events_meg_, event_id, tmin=tmin, tmax=tmax,
                preload=True, baseline=(None, 0))

for i in np.arange(20):
    plt.plot(data_photodiod[i,:])
    
# check timing photodiod
channel_photodiod = np.where(np.array(epochs.ch_names) == 'UADC016-2104')[0][0]
data_photodiod = epochs._data[:, channel_photodiod, :]
fig, ax = plt.subplots(1, figsize=[10, 4])
ax.matshow(data_photodiod, aspect='auto', cmap='gray',
           extent=[np.min(epochs.times), np.max(epochs.times),
                   0, len(epochs)])
for ii in np.arange(-.200, .300, .100):
    ax.axvline(ii)
ax.set_xticks(np.arange(-.100, 1.9, .100))
ax.set_xlabel('Times')
ax.set_ylabel('Trials')
plt.show()

events_meg_[:, 1] = run_number
events_meg.append(events_meg_)

epochs_list.append(epochs)

events_meg = np.vstack(events_meg)  # concatenate all meg events
epochs = concatenate_epochs(epochs_list)

# Save
fname = op.join(path_raw, 'epochs.fif')
epochs.save(fname)

#Fig trace
evoked = epochs.average()
evoked.plot()

#Fig Topo
evoked.plot_topomap()

#Fig Topo. Manually display time of the topo
times = np.arange(0.05, 0.151, 0.05)
evoked.plot_topomap(times=times, ch_type='mag')

#Fig Topo. Automatically select the peacks
evoked.plot_topomap(times='peaks', ch_type='mag')

#Visualize epoch data
epochs.plot_image(97)



######################################################################


# Pre-processing
raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
raw.filter(0.75, 60)

# start, stop = raw.time_as_index([100, 115])
events = find_events(raw, stim_channel='UPPT001')

event_id = dict (NoGo=32,Go=1) #

epochs = Epochs(raw, events, event_id,
                tmin=-0.1, tmax=0.5, preload=True,
                baseline=[-0.1, 0])

evoked = epochs.average()
evoked.plot()

#Fig test 1
#fig.tight_layout()
#picks = mne.pick_types(evoked.info, meg=True, eeg=False, eog=False)
#evoked.plot(spatial_colors=True, gfp=True, picks=picks)

#Fig Topo
evoked.plot_topomap()

#Fig Topo. Manually display time of the topo
times = np.arange(0.05, 0.151, 0.05)
evoked.plot_topomap(times=times, ch_type='mag')

#Fig Topo. Automatically select the peacks
evoked.plot_topomap(times='peaks', ch_type='mag')

#Fig test
epochs.plot_image(97)
epochs.plot_topo_image(vmin=-200, vmax=200, title='ERF images')

#n_cycles = 2
#frequencies = np.arange(7, 30, 3)
#Fs = raw.info['sfreq']

#data = epochs.get_data()
#from mne.time_frequency import induced_power
#power, phase_lock = induced_power(data, Fs=Fs, frequencies=frequencies,
#                                  n_cycles=2, n_jobs=1)
