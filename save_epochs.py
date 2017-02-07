import numpy as np
import os.path as op
import mne
from mne import Epochs, find_events, pick_channels
from mne.io import read_raw_ctf
import matplotlib.pyplot as plt
from path import path
from mne.epochs import concatenate_epochs


path_data = '/Users/quentinra/Desktop/MEGdata/Etiennedata'
subject = '20160803'
task = 'gng'



path_raw = op.join(path_data, subject, task)
runs = path(path_raw).glob('*.ds')

if len(runs) != 8:
    raise RuntimeError('Could not find 8 runs in %s' % path_raw)

events_meg = list()
epochs_list = list()

for run_number, this_run in enumerate(runs):

    fname_raw = op.join(path_raw, this_run)
    raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
    events_meg_ = mne.find_events(raw)

    raw.filter(l_freq=.1, h_freq=60.0, l_trans_bandwidth=.01)
    tmin = -.200
    tmax = 1.500

    occ = pick_channels(Occpar_list, include=[])

    epochs = Epochs(raw, events_meg_, event_id=[1, 32],
                    tmin=tmin, tmax=tmax, preload=True, picks=Occpar_list,
                    baseline=(None, 0))

    events_meg_[:, 1] = run_number
    events_meg.append(events_meg_)

    epochs_list.append(epochs)

events_meg = np.vstack(events_meg)  # concatenate all meg events
epochs = concatenate_epochs(epochs_list)

# anonymize
# anonymize(epochs)

# Save
fname = op.join(path_raw, 'epochs.fif')
epochs.save(fname)

evoked = epochs.average()
evoked.plot()
