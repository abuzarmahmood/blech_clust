import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve, find_peaks
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import utils.spike_template as st
from tqdm import tqdm

def get_filtered_electrode(data, freq = [300.0, 3000.0], sampling_rate = 30000.0):
		el = 0.195*(data)
		m, n = butter(2, [2.0*freq[0]/sampling_rate, 2.0*freq[1]/sampling_rate], btype = 'bandpass') 
		filt_el = filtfilt(m, n, el)
		return filt_el

def extract_waveforms_abu(filt_el, spike_snapshot = [0.5, 1.0], 
								    sampling_rate = 30000.0,
								    threshold_mult = 5.0):

		m = np.mean(filt_el)
		th = threshold_mult*np.median(np.abs(filt_el)/0.6745)

		negative = np.where(filt_el <= m-th)[0] 
		positive = np.where(filt_el >= m+th)[0] 
		# Marking breaks in detected threshold crossings 
		neg_changes = np.concatenate(([0],np.where(np.diff(negative) > 1)[0]+1))
		pos_changes = np.concatenate(([0],np.where(np.diff(positive) > 1)[0]+1))
		
		# Mark indices to be extracted
		neg_inds = [(negative[neg_changes[x]],negative[neg_changes[x+1]-1]) \
				for x in range(len(neg_changes)-1)]
		pos_inds = [(positive[pos_changes[x]],positive[pos_changes[x+1]-1]) \
				for x in range(len(pos_changes)-1)]

		# Mark the extremum of every threshold crossing
		minima = [np.argmin(filt_el[start:(end+1)]) + start \
				for start,end in neg_inds]
		maxima = [np.argmax(filt_el[start:(end+1)]) + start \
				for start,end in pos_inds]

		polarity = np.concatenate(([-1]*len(minima),[1]*len(maxima)))

		spike_times = np.concatenate((minima,maxima))

		needed_before = int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))
		needed_after = int((spike_snapshot[1]+ 0.1)*(sampling_rate/1000.0))
		before_inds = spike_times - needed_before
		after_inds = spike_times + needed_after

		# Make sure event has required window around it
		relevant_inds = (before_inds > 0) * (after_inds < len(filt_el))
		before_inds = before_inds[relevant_inds]
		after_inds = after_inds[relevant_inds]
		slices = np.array([filt_el[start:end] \
				for start,end in zip(before_inds,after_inds)])

		return slices, spike_times[relevant_inds], polarity[relevant_inds], m, th

def extract_waveforms_hannah(filt_el, dir_name, spike_snapshot = [0.5, 1.0], 
								    sampling_rate = 30000.0,
								    threshold_mult = 5.0,
									cut_percentile = 25):
		print('Pulling peak times.')
		#Sliding thresholding
		print('\t Calculating Threshold')
		len_filt_el = len(filt_el)
		sec_samples = int(60*5*sampling_rate) #2.5 minutes in samples
		start_times = np.arange(0,len_filt_el-sec_samples,sec_samples)
		threshold_vals = []
		for s_i in tqdm(range(len(start_times))):
			s_t = start_times[s_i]
			filt_el_clip = np.array(filt_el)[max(s_t,0):min(s_t+sec_samples,len_filt_el)]
			th_clip = threshold_mult*np.median(np.abs(filt_el_clip)/0.6745)
			threshold_vals.extend([th_clip])
		#Percentile mean and threshold values
		m = 0 #Assuming data cleaning worked properly
		th = min(threshold_vals)
		print('\t Selected thresh = ' + str(round(th,3)))
		#Find peaks crossing threshold in either direction and combine
		peak_dist_min = np.ceil((1/1000)*sampling_rate) #Peaks must be at least 1 ms apart
		minima = np.array(find_peaks(-1*(filt_el),height=th)[0]) #indices of - peaks
		maxima = np.array(find_peaks(filt_el,height=th)[0]) #indices of + peaks
		#Separately template match minima
		print('\t Template sorting negative spikes')
		#Set snippet parameters
		needed_before = int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))
		needed_after = int((spike_snapshot[1]+ 0.1)*(sampling_rate/1000.0))
		#Grab positive and negative spike waveforms and 'polarity'
		before_inds = minima - needed_before
		after_inds = minima + needed_after
		relevant_inds = (before_inds > 0) * (after_inds < len(filt_el))
		before_inds = before_inds[relevant_inds]
		after_inds = after_inds[relevant_inds]
		minima = minima[relevant_inds]
		minima_slices = np.array([filt_el[start:end] \
				for start,end in zip(before_inds,after_inds)])
		#Returns only the template-match thresholded slices
		minima_slices, relevant_inds = st.spike_template_sort(minima_slices,'min',sampling_rate,needed_before,needed_after,
								cut_percentile,dir_name)
		minima = minima[relevant_inds]
		
		#Separately threshold maxima
		print('\t Template sorting positive spikes')
		#Set snippet parameters
		needed_before = int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))
		needed_after = int((spike_snapshot[1]+ 0.1)*(sampling_rate/1000.0))
		#Grab positive and negative spike waveforms and 'polarity'
		before_inds = maxima - needed_before
		after_inds = maxima + needed_after
		relevant_inds = (before_inds > 0) * (after_inds < len(filt_el))
		before_inds = before_inds[relevant_inds]
		after_inds = after_inds[relevant_inds]
		maxima = maxima[relevant_inds]
		maxima_slices = np.array([filt_el[start:end] \
				for start,end in zip(before_inds,after_inds)])
		#Returns only the template-match thresholded slices
		maxima_slices, relevant_inds = st.spike_template_sort(maxima_slices,'max',sampling_rate,needed_before,needed_after,
								cut_percentile,dir_name)
		maxima = maxima[relevant_inds]
		
		#Combine thresholded results
		slices = np.concatenate([minima_slices,maxima_slices])
		spike_times = np.concatenate((minima,maxima))
		polarity = np.concatenate(([-1]*len(minima),[1]*len(maxima)))
		print('\t Total number of peaks = ' + str(len(spike_times)))

		return slices, spike_times, polarity, m, th

def extract_waveforms(filt_el, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
		m = np.mean(filt_el)
		th = 5.0*np.median(np.abs(filt_el)/0.6745)
		#pos = np.where(filt_el <= m-th)[0]
		pos = np.where( (filt_el <= m-th) | (filt_el > m+th) )[0]
		
		changes = []
		for i in range(len(pos)-1):
				if pos[i+1] - pos[i] > 1:
						changes.append(i+1)

		# slices = np.zeros((len(changes)-1,150))

		slices = []
		spike_times = []
		for i in range(len(changes) - 1):
				minimum = np.where(filt_el[pos[changes[i]:changes[i+1]]] == \
						np.min(filt_el[pos[changes[i]:changes[i+1]]]))[0]

				#print minimum, len(slices), len(changes), len(filt_el)
				# try slicing out the putative waveform, 
				# only do this if there are 10ms of data points 
				# (waveform is not too close to the start or end of the recording)
				if pos[minimum[0]+changes[i]] \
						- int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))> 0 \
						and pos[minimum[0]+changes[i]] + int((spike_snapshot[1]+ 0.1)\
								    *(sampling_rate/1000.0)) < len(filt_el):
						slices.append(filt_el[pos[minimum[0]+changes[i]] - \
								int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0)) : \
										    pos[minimum[0]+changes[i]] + int((spike_snapshot[1] \
												    + 0.1)*(sampling_rate/1000.0))])
						spike_times.append(pos[minimum[0]+changes[i]])

		return np.array(slices), spike_times, m, th

def dejitter(slices, spike_times, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
		x = np.arange(0,len(slices[0]),1)
		# Support vector for 10x interpolation
		xnew = np.arange(0,len(slices[0])-1,0.1)

		# Calculate the number of samples to be sliced out around each spike's minimum
		before = int((sampling_rate/1000.0)*(spike_snapshot[0]))
		after = int((sampling_rate/1000.0)*(spike_snapshot[1]))
		
		#slices_dejittered = []
		slices_dejittered = np.zeros((slices.shape[0],(before+after)*10))
		spike_times_dejittered = []
		for i in range(len(slices)):
				f = interp1d(x, slices[i])
				# 10-fold interpolated spike
				ynew = f(xnew)
				# Find minimum only around the center of the waveform
				# Since we're slicing around the waveform as well
				minimum = np.where(ynew == np.min(ynew))[0][0]
				# Only accept spikes if the interpolated minimum has 
				# shifted by less than 1/10th of a ms 
				# (3 samples for a 30kHz recording, 
				# 30 samples after interpolation)
				# If minimum hasn't shifted at all, 
				# then minimum - 5ms should be equal to zero 
				# (because we sliced out 5 ms before the minimum 
				# in extract_waveforms())
				# We use this property in the if statement below
				cond1 = np.abs(minimum - \
						int((spike_snapshot[0] +0.1)\
						*(sampling_rate/100.0))) \
						    <= int(10.0*(sampling_rate/10000.0))
				y_temp = ynew[minimum - before*10 : minimum + after*10]
				# Or slices which (SOMEHOW) don't have the expected size
				cond2 = len(y_temp) == slices_dejittered.shape[-1] 
				if cond1 and cond2:
						#slices_dejittered.append(ynew[minimum - before*10 : minimum + after*10])
						slices_dejittered[i] = y_temp 
						spike_times_dejittered.append(spike_times[i])

		# Remove placeholder for slices which didnt meet the condition criteria
		slices_dejittered = \
		slices_dejittered[np.sum(slices_dejittered,axis=-1) != 0]

		return slices_dejittered, np.array(spike_times_dejittered)

def dejitter_abu3(slices, 
				spike_times, 
				polarity,
				spike_snapshot = [0.5, 1.0], 
				sampling_rate = 30000.0):

		"""
		Dejitter without interpolation and see what breaks :P
		"""
		# Calculate the number of samples to be sliced 
		#out around each spike's minimum
		before = int((sampling_rate/1000.0)*(spike_snapshot[0]))
		after = int((sampling_rate/1000.0)*(spike_snapshot[1]))

		# Determine positive or negative spike and flip
		# positive spikes so everything is aligned by minimum
		# Then flip positive spikes back to being positive
		flipped_slices = np.copy(slices)
		flipped_slices[polarity > 0] *= -1

		#interp_slices = np.array([interp1d(x, this_slice)(xnew) \
		#		for this_slice in flipped_slices])

		# Cut out part around focus of spike snapshot to use
		# for finding minima
		# 3 bins (0.1 ms) is the wiggle room we gave ourselves 
		# when extracting spikes, therefore, each spike is organized as
		#   0.1 ms |-| Before |-| Minimum |-| After |-| 0.1 ms
		# We will use 0.1ms around the minimum to dejitter the spike 
		cut_radius = 3
		cut_tuple = (int((before) + (cut_radius/2)), 
				int(flipped_slices.shape[1] - (after) - (cut_radius/2)))
		# minima will tell us how much each spike needs to be shifted
		minima = np.argmin(flipped_slices[:,cut_tuple[0]:cut_tuple[1]],
						axis=-1) + (before) + (cut_radius/2) 

		# Extract windows AROUND minima
		slices_dejittered = np.array([this_slice[\
		    int(this_min - (before)) : int(this_min + (after))] \
		    for this_slice, this_min in zip(flipped_slices, minima)])

		# Flip positive slices
		slices_dejittered[polarity > 0] *= -1

		return slices_dejittered, spike_times

def scale_waveforms(slices_dejittered):
		energy = np.sqrt(np.sum(slices_dejittered**2, axis = 1))/len(slices_dejittered[0])
		scaled_slices = np.zeros((len(slices_dejittered),len(slices_dejittered[0])))
		for i in range(len(slices_dejittered)):
				scaled_slices[i] = slices_dejittered[i]/energy[i]

		return scaled_slices, energy

def implement_pca(scaled_slices):
		pca = PCA()
		pca_slices = pca.fit_transform(scaled_slices)   
		return pca_slices, pca.explained_variance_ratio_

def clusterKMeans(data, n_clusters, n_iter, restarts, threshold):
    kmeans = KMeans(n_clusters = n_clusters, 
				    n_init = restarts,
				    max_iter = n_iter,
				    tol = threshold).fit(data)
    return kmeans.labels_

def clusterGMM(data, n_clusters, n_iter, restarts, threshold):
		

		g = []
		bayesian = []

		for i in range(restarts):
				g.append(GaussianMixture(n_components = n_clusters, covariance_type = 'full', tol = threshold, random_state = i, max_iter = n_iter))
				#g.append(GaussianMixture(n_components = n_clusters,
				#    covariance_type = 'diag', tol = threshold, random_state = i, max_iter = n_iter))
				g[-1].fit(data)
				if g[-1].converged_:
						bayesian.append(g[-1].bic(data))
				else:
						del g[-1]

		#print len(akaike)
		bayesian = np.array(bayesian)
		best_fit = np.where(bayesian == np.min(bayesian))[0][0]
		
		predictions = g[best_fit].predict(data)

		return g[best_fit], predictions, np.min(bayesian)

