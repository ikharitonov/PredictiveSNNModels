import numpy as np
import h5py

def get_digit_inds(train_file_path, speaker_id=0, digit=0):
    train_file = h5py.File(train_file_path, 'r')
    
    speaker_digit_ids = np.array(train_file['extra']['speaker'])

    inds1 = np.where(speaker_digit_ids == speaker_id)

    digit_labels = np.array(train_file['labels'])
    digit_labels[inds1]

    inds2 = np.where(digit_labels[inds1] < 10) # only english 0-9

    speaker_english_digits_ids = inds1[0][inds2]

    inds3 = np.where(train_file['labels'][speaker_english_digits_ids] == digit)
    digits_inds = speaker_english_digits_ids[inds3]

    return digits_inds

def get_digit_data(train_file_path, indices):
    train_file = h5py.File(train_file_path, 'r')
    trials_units = train_file['spikes']['units'][indices]
    trials_times = train_file['spikes']['times'][indices]
    
    return trials_units, trials_times, len(trials_units)

def trim_to_length(units_arr, times_arr, time_length):
    # Trims all format 1 arrays for all trials contained to specified length
    temp_units_arr, temp_times_arr = [], []
    for i in range(len(units_arr)):
        inds = np.where(times_arr[i] >= time_length)
        if len(inds[0]) == 0:
            temp_units_arr.append(units_arr[i])
            temp_times_arr.append(times_arr[i])
        else:
            temp_units_arr.append(units_arr[i][:inds[0][0]])
            temp_times_arr.append(times_arr[i][:inds[0][0]])
    
    return temp_units_arr, temp_times_arr

def get_time_array(duration, frequency):
    return np.linspace(0,duration,int(frequency*duration))

def resample(units_arr, times_arr, duration, frequency, pad_length=None):
    # Resamples and pads spike trains
    
    out_t, out_spikes = [], []
    
    for trial_id in range(len(units_arr)):
        t = get_time_array(duration, frequency)
        all_units_list = np.unique(units_arr[trial_id])
        spikes = np.zeros((t.shape[0], all_units_list.shape[0]))

        for i, spike in enumerate(times_arr[trial_id]):
            t_ind = np.where(t <= spike)[0][-1] # assign the spike to the closest smaller time bin
            u_ind = np.where(all_units_list==units_arr[trial_id][i])

            spikes[t_ind, u_ind] = 1

        if pad_length:
            if spikes.shape[1] > pad_length:
                spikes = spikes[:, :pad_length]
            elif spikes.shape[1] < pad_length:
                diff = pad_length - spikes.shape[1]
                spikes = np.concatenate((spikes, np.zeros((t.shape[0], diff))), axis=1)
        
        out_t.append(t)
        out_spikes.append(spikes)

    return np.array(out_t), np.array(out_spikes)

def single_trial_plt_conversion(param1, param2, data_format=1):
    spike_data = []
    if data_format==1:
        trial_units = param1
        trial_times = param2
        for unit_id in np.unique(trial_units):
            inds = np.where(trial_units == unit_id)
            spike_data.append(list(trial_times[inds]))
    elif data_format==2:
        t = param1
        spikes = param2
        spike_data = [t[np.where(spikes[:, u_id]==1)] for u_id in range(spikes.shape[1])]
    return spike_data

def get_format_2_data(train_file_path, speaker_id, digit, duration, resampling_frequency, unit_padding_length):
    inds = get_digit_inds(train_file_path, speaker_id=speaker_id, digit=digit)
    units, times, trials_num = get_digit_data(train_file_path, inds)
    trimmed_units, trimmed_times = trim_to_length(units, times, duration)
    t, spikes = resample(trimmed_units, trimmed_times, duration, resampling_frequency, pad_length=unit_padding_length)
    return t, spikes