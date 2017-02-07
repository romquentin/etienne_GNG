##############################################################################
############################### Resting state ################################
##############################################################################

import numpy as np
import os.path as op

import mne
from mne import Epochs, find_events
from mne.io import read_raw_ctf
import matplotlib.pyplot as plt

from mne.preprocessing import maxwell_filter

############################### PRE PROCESSING ###############################

# Find the folder with MEG data
data_path = '/Users/Sallardef/Downloads/MEG-DATA/'
subject = '20160928'
run = 'TCBTDAUP_rest_20160928_01.ds'

fname_raw = op.join(data_path, subject, run)
print(fname_raw)

raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')


# Display the name of the trigger with the time
events = find_events(raw, stim_channel='UPPT001')

# Let's check if there is noise and if yes in which frequencies
raw.plot_psd(area_mode='range', tmax=10.0, fmax=200)
raw.plot(duration=30, start=180, n_channels=100, remove_dc=True)

# Pre processing. Filter
raw.filter(l_freq=1, h_freq=50.0)

# Let's check if the noise is reduced - 2 graph
raw.plot_psd(area_mode='range', tmax=10.0, fmax=200)
raw.plot(duration=30, start=180, n_channels=100, remove_dc=True)



# Marking bad raw segments with annotations

raw_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(events)
# Center to cover the whole blink with full duration of 0.5s:
onset = events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                  orig_time=raw.info['meas_date'])

raw.plot(events=events, duration=30, start=0)  # To see the annotated segments.








################################## EPOCHING ##################################

## start, stop = raw.time_as_index([100, 115])

# set first event to 30sec (18000 in time frame)
events[0, 0] = 18000

# create fake events of 1sec during the entire time of resting state
# index of the first event (start of the resting state)
start = events[0, 0]

# index of the last one (end of the resting state at 4.30min)
stop = raw.time_as_index(270)

# time between each fake events as index
step = raw.time_as_index(1)

nb_events = (stop - start)/step

# Loop to create events
events_time = start
fake_events = events
for i in xrange(nb_events-1):
    print i
    events_time = events_time + step
    print events_time
    fake_events = np.append(fake_events, [[int(events_time), 0, 64
                                           ]], axis=0)

# Event (epoch) window 2sec with overlapping 1sec
epochs = Epochs(raw, fake_events, event_id=None,
                tmin=-1, tmax=1, preload=True,
                baseline=None)


############################# ARTIFACT REJECTION #############################

#Artifact rejection using peak-to-peak amplitude and flat signal detection
#Value used from the MNE tutorial. No idea about the value to use
reject = dict (grad=4000e-13)
flat = dict (grad=1e-10)

# Reject the bad epochs
epochs.drop_bad()

raw.plot(epochs, duration=30, start=30)

############################### VISUALIZATION ################################

#Visualize epoch concatenate
epochs.plot(block=True)

#Visualize epoch data
epochs.plot_image(97)


raw.plot(order='position')

evoked = mne.read_evokeds(epochs, baseline=(None, 0), proj=True)
print(evoked)




#Compute resting state by averaging and plot it:
evoked = epochs.average()
print(evoked)
evoked.plot()

epochs.plot_psd(fmin=2., fmax=40.)


print(epochs.drop_log)
epochs.plot_drop_log


evoked.plot(exlude=[])






#####################################################
############## Compute phase lag index ##############
#####################################################

from scipy import linalg
from mne.connectivity import spectral_connectivity

print(__doc__)

# Compute connectivity for band containing the evoked response.
# We exclude the baseline period
fmin, fmax = 3., 9.
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

# con is a 3D array where the last dimension is size one since we averaged
# over frequencies in a single band. Here we make it 2D
con = con[:, :, 0]

# Now, visualize the connectivity in 3D
from mayavi import mlab  # noqa

mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))



# Plot the sensor locations






#####################################################
############## DO NOT WORK !!!!!!!!!!! ##############
#####################################################


# Check out freq/power spectral density on all channel types
# by averaging across epochs
epochs.plot_psd(fmin=2., fmax=40.)

# Have a look on the spatial distribution of the PSD (power spectral density)
epochs.plot_psd_topomap(ch_type='mag', normalize=True, #vmin=0.1, vmax=0.4,
                        bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'),
                        (12, 30, 'Beta')])


#Inspect power
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet

freqs = np.arange(6, 30, 3)  # define frequencies of interest
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
power.plot([82], baseline=(-0.5, 0), mode='logratio')

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='mag', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0],
                   title='Alpha', vmax=0.45, show=False)
power.plot_topomap(ch_type='mag', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1],
                   title='Beta', vmax=0.45, show=False)
mne.viz.tight_layout()
plt.show()

##evoked = epochs.average()
##evoked.plot()

##epochs.plot_image(97)

##n_cycles = 2
##frequencies = np.arange(7, 30, 3)
##Fs = raw.info['sfreq']

##data = epochs.get_data()
##from mne.time_frequency import induced_power
##power, phase_lock = induced_power(data, Fs=Fs, frequencies=frequencies, n_cycles=2, n_jobs=1)
