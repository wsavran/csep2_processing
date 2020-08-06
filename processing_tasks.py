import os
from collections import defaultdict

import numpy
from matplotlib import pyplot as plt

from csep.utils.basic_types import seq_iter, AdaptiveHistogram
from csep.utils.calc import _compute_likelihood, bin1d_vec, _compute_spatial_statistic
from csep.utils.constants import CSEP_MW_BINS, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_WEEK
from csep.models import EvaluationResult
from csep.core.repositories import FileSystem
from csep.utils.plots import plot_number_test, plot_magnitude_test, plot_likelihood_test, plot_spatial_test, \
    plot_cumulative_events_versus_time_dev, plot_magnitude_histogram_dev, plot_distribution_test, plot_probability_test, \
    plot_spatial_dataset
from csep.utils.stats import get_quantiles, cumulative_square_diff, sup_dist

# todo: refactor these methods to not perform any filtering of catalogs inside the processing task

class AbstractProcessingTask:
    def __init__(self, data=None, name=None, min_mw=2.5, n_cat=None, mws=None):
        self.data = data or []
        # to-be deprecated
        self.mws = mws or [2.5, 3.0, 3.5, 4.0, 4.5]
        self.min_mw = min_mw
        self.n_cat = n_cat
        self.name = name
        self.ax = []
        self.fnames = []
        self.needs_two_passes = False
        self.buffer = []
        self.region = None
        self.buffer_fname = None
        self.fhandle = None
        self.archive = True
        self.version = 1

    @staticmethod
    def _build_filename(dir, mw, plot_id):
        basename = f"{plot_id}_mw_{str(mw).replace('.','p')}".lower()
        return os.path.join(dir, basename)

    def process(self, data):
        raise NotImplementedError('must implement process()!')

    def process_again(self, catalog, args=()):
        """ This function defaults to pass unless the method needs to read through the data twice. """
        pass

    def post_process(self, obs, args=None):
        """
        Compute evaluation of data stored in self.data.

        Args:
            obs (csep.Catalog): used to evaluate the forecast
            args (tuple): args for this function

        Returns:
            result (csep.core.evaluations.EvaluationResult):

        """
        result = EvaluationResult()
        return result

    def plot(self, results, plot_dir, show=False):
        """
        plots function, typically just a wrapper to function in utils.plotting()

        Args:
            show (bool): show plot, if plotting multiple, just run on last.
            filename (str): where to save the file
            plot_args (dict): plotting args to pass to function

        Returns:
            axes (matplotlib.axes)

        """
        raise NotImplementedError('must implement plot()!')

    def store_results(self, results, dir):
        """
        Saves evaluation results serialized into json format. This format is used to recreate the results class which
        can then be plotted if desired. The following directory structure will be created:

        | dir
        |-- n-test
        |---- n-test_mw_2.5.json
        |---- n_test_mw_3.0.json
        |-- m-test
        |---- m_test_mw_2.5.json
        |---- m_test_mw_3.0.json
        ...

        The results iterable should only contain results for a single evaluation. Typically they would contain different
        minimum magnitudes.

        Args:
            results (Iterable of EvaluationResult): iterable object containing evaluation results. this could be a list or tuple of lists as well
            dir (str): directory to store the testing results. name will be constructed programatically.

        Returns:
            None

        """
        success = False
        if self.archive == False:
            return

        # handle if results is just a single result
        if isinstance(results, EvaluationResult):
            repo = FileSystem(url=self._build_filename(dir, results.min_mw, results.name) + '.json')
            if repo.save(results.to_dict()):
                success = True
            return success
        # or if its an iterable
        for idx in seq_iter(results):
            # for debugging
            if isinstance(results[idx], tuple) or isinstance(results[idx], list):
                result = results[idx]
            else:
                result = [results[idx]]
            for r in result:
                repo = FileSystem(url=self._build_filename(dir, r.min_mw, r.name) + '.json')
                if repo.save(r.to_dict()):
                    success = True
        return success

    def store_data(self, dir):
        """ Store the intermediate data used to calculate the results for the evaluations. """
        raise NotImplementedError


class NumberTest(AbstractProcessingTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mws = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    def process(self, catalog, filter=False):
        if not self.name:
            self.name = catalog.name
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            counts.append(cat_filt.event_count)
        self.data.append(counts)

    def post_process(self, obs, args=None):
        # we dont need args for this function
        _ = args
        results = {}
        data = numpy.array(self.data)
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            observation_count = obs_filt.event_count
            # get delta_1 and delta_2 values
            delta_1, delta_2 = get_quantiles(data[:,i], observation_count)
            # prepare result
            result = EvaluationResult(test_distribution=data[:,i],
                          name='N-Test',
                          observed_statistic=observation_count,
                          quantile=(delta_1, delta_2),
                          status='Normal',
                          obs_catalog_repr=obs.date_accessed,
                          sim_name=self.name,
                          min_mw=mw,
                          obs_name=obs.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):

        for mw, result in results.items():
            # compute bin counts, this one is special because of integer values
            td = result.test_distribution
            min_bin, max_bin = numpy.min(td), numpy.max(td)
            # hard-code some logic for bin size
            bins = numpy.arange(min_bin, max_bin)
            if len(bins) == 1:
                bins = 3
            n_test_fname = AbstractProcessingTask._build_filename(plot_dir, mw, 'n_test')
            _ = plot_number_test(result, show=show, plot_args={'percentile': 95,
                                                                'title': f'Number Test, M{mw}+',
                                                                'bins': bins,
                                                                'filename': n_test_fname})
            self.fnames.append(n_test_fname)


class MagnitudeTest(AbstractProcessingTask):

    def __init__(self, mag_bins=None, **kwargs):
        super().__init__(**kwargs)
        self.mws = [2.5, 3.0, 3.5, 4.0]
        self.mag_bins = mag_bins
        self.version = 4

    def process(self, catalog):
        if not self.name:
            self.name = catalog.name
        # magnitude mag_bins should probably be bound to the region, although we should have a SpaceMagnitudeRegion class
        if self.mag_bins is None:
            try:
                self.mag_bins = catalog.region.mag_bins
            except:
                self.mag_bins = CSEP_MW_BINS
        # optimization idea: always compute this for the lowest magnitude, above this is redundant
        mags = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            binned_mags = cat_filt.magnitude_counts(mag_bins=self.mag_bins)
            mags.append(binned_mags)
        # data shape (n_cat, n_mw, n_mw_bins)
        self.data.append(mags)

    def post_process(self, obs, args=None):
        # we dont need args
        _ = args
        results = {}
        for i, mw in enumerate(self.mws):
            test_distribution = []
            # get observed magnitude counts
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            if obs_filt.event_count == 0:
                print(f"Skipping {mw} in Magnitude test because no observed events.")
                continue
            obs_histogram = obs_filt.magnitude_counts(mag_bins=self.mag_bins)
            n_obs_events = numpy.sum(obs_histogram)
            mag_counts_all = numpy.array(self.data)
            # get the union histogram, simply the sum over all catalogs, (n_cat, n_mw)
            union_histogram = numpy.sum(mag_counts_all[:,i,:], axis=0)
            n_union_events = numpy.sum(union_histogram)
            union_scale = n_obs_events / n_union_events
            scaled_union_histogram = union_histogram * union_scale
            for j in range(mag_counts_all.shape[0]):
                n_events = numpy.sum(mag_counts_all[j,i,:])
                if n_events == 0:
                    continue
                scale = n_obs_events / n_events
                catalog_histogram = mag_counts_all[j,i,:] * scale

                test_distribution.append(cumulative_square_diff(numpy.log10(catalog_histogram+1), numpy.log10(scaled_union_histogram+1)))
            # compute statistic from the observation
            obs_d_statistic = cumulative_square_diff(numpy.log10(obs_histogram+1), numpy.log10(scaled_union_histogram+1))
            # score evaluation
            _, quantile = get_quantiles(test_distribution, obs_d_statistic)
            # prepare result
            result = EvaluationResult(test_distribution=test_distribution,
                                      name='M-Test',
                                      observed_statistic=obs_d_statistic,
                                      quantile=quantile,
                                      status='Normal',
                                      min_mw=mw,
                                      obs_catalog_repr=obs.date_accessed,
                                      obs_name=obs.name,
                                      sim_name=self.name)
            results[mw] = result
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # get the filename
        for mw, result in results.items():
            m_test_fname = self._build_filename(plot_dir, mw, 'm-test')
            plot_args = {'percentile': 95,
                         'title': f'Magnitude Test, M{mw}+',
                         'bins': 'auto',
                         'filename': m_test_fname}
            _ = plot_magnitude_test(result, show=False, plot_args=plot_args)
            self.fnames.append(m_test_fname)

    def _build_filename(self, dir, mw, plot_id):
        try:
            mag_dh = self.mag_bins[1] - self.mag_bins[0]
            mag_dh_str = f"_dmag{mag_dh:.1f}".replace('.','p').lower()
        except:
            mag_dh_str = ''
        basename = f"{plot_id}_mw_{str(mw).replace('.', 'p')}{mag_dh_str}".lower()
        return os.path.join(dir, basename)


class LikelihoodAndSpatialTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region = None
        self.test_distribution_spatial = []
        self.test_distribution_likelihood = []
        self.cat_id = 0
        self.needs_two_passes = True
        self.buffer = []
        self.fnames = {}
        self.fnames['l-test'] = []
        self.fnames['s-test'] = []
        self.version = 5

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        # compute stuff from data
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_counts = cat_filt.spatial_counts()
            counts.append(gridded_counts)
        # we want to aggregate the counts in each bin to preserve memory
        if len(self.data) == 0:
            self.data = numpy.array(counts)
        else:
            self.data += numpy.array(counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the data
        time_horizon, n_cat, end_epoch, obs = args
        apprx_rate_density = numpy.array(self.data) / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1)

        # unfortunately, we need to iterate twice through the catalogs for this, unless we start pre-processing
        # everything and storing approximate cell-wise rates
        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_cat = cat_filt.spatial_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm
        self.test_distribution_likelihood.append(lhs)
        self.test_distribution_spatial.append(lhs_norm)

    def post_process(self, obs, args=None):
        cata_iter, time_horizon, end_epoch, n_cat = args
        results = {}

        apprx_rate_density = numpy.array(self.data) / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1)

        test_distribution_likelihood = numpy.array(self.test_distribution_likelihood)
        # there can be nans in the spatial distribution
        test_distribution_spatial = numpy.array(self.test_distribution_spatial)
        # prepare results for each mw
        for i, mw in enumerate(self.mws):
            # get observed likelihood
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            if obs_filt.event_count == 0:
                print(f'Skipping pseudo-likelihood based tests for M{mw}+ because no events in observed observed_catalog.')
                continue
            n_obs = obs_filt.get_number_of_events()
            gridded_obs = obs_filt.spatial_counts()
            obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
            message = "normal"
            if obs_lh == -numpy.inf or obs_lh_norm == -numpy.inf:
                idx_good_sim = apprx_rate_density[i,:] != 0
                new_gridded_obs = gridded_obs[idx_good_sim]
                new_n_obs = numpy.sum(new_gridded_obs)
                print(f"Found -inf as the observed likelihood score for M{self.mws[i]}+. "
                      f"Assuming event(s) occurred in undersampled region of forecast.\n"
                      f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")
                if new_n_obs == 0:
                    print(f'Skipping pseudo-likelihood based tests for M{mw}+ because no events in observed observed_catalog '
                          f'after correcting for under-sampling in forecast.')
                    continue
                new_ard = apprx_rate_density[i,idx_good_sim]
                # we need to use the old n_obs here, because if we normalize the ard to a different value the observed
                # statistic will not be computed correctly.
                obs_lh, obs_lh_norm = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count[i], n_obs)
                message = "undersampled"

            # determine outcome of evaluation, check for infinity
            _, quantile_likelihood = get_quantiles(test_distribution_likelihood[:,i], obs_lh)

            # build evaluation result
            result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood[:,i],
                                                 name='L-Test',
                                                 observed_statistic=obs_lh,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 min_mw=mw,
                                                 obs_catalog_repr=obs.date_accessed,
                                                 sim_name=self.name,
                                                 obs_name=obs.name)

            # check for nans here
            test_distribution_spatial_1d = test_distribution_spatial[:,i]
            if numpy.isnan(numpy.sum(test_distribution_spatial_1d)):
                test_distribution_spatial_1d = test_distribution_spatial_1d[~numpy.isnan(test_distribution_spatial_1d)]

            if n_obs == 0 or numpy.isnan(obs_lh_norm):
                message = "not-valid"
                quantile_spatial = -1
            else:
                _, quantile_spatial = get_quantiles(test_distribution_spatial_1d, obs_lh_norm)

            result_spatial = EvaluationResult(test_distribution=test_distribution_spatial_1d,
                                          name='S-Test',
                                          observed_statistic=obs_lh_norm,
                                          quantile=quantile_spatial,
                                          status=message,
                                          min_mw=mw,
                                          obs_catalog_repr=obs.date_accessed,
                                          sim_name=self.name,
                                          obs_name=obs.name)

            results[mw] = (result_likelihood, result_spatial)

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result_tuple in results.items():
            # plot likelihood test
            l_test_fname = self._build_filename(plot_dir, mw, 'l-test')
            plot_args = {'percentile': 95,
                         'title': f'Pseudo-Likelihood Test, M{mw}+',
                         'filename': l_test_fname}
            _ = plot_likelihood_test(result_tuple[0], axes=None, plot_args=plot_args, show=show)

            # we can access this in the main program if needed
            # self.ax.append((ax, spatial_ax))
            self.fnames['l-test'].append(l_test_fname)

            if result_tuple[1].status == 'not-valid':
                print(f'Skipping plot for spatial test on {mw}. Test results are not valid, likely because no earthquakes observed in target observed_catalog.')
                continue

            # plot spatial test
            s_test_fname = self._build_filename(plot_dir, mw, 's-test')
            plot_args = {'percentile': 95,
                         'title': f'Spatial Test, M{mw}+',
                         'filename': s_test_fname}
            _ = plot_spatial_test(result_tuple[1], axes=None, plot_args=plot_args, show=False)
            self.fnames['s-test'].append(s_test_fname)


class SpatialTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region = None
        self.test_distribution_spatial = []
        self.cat_id = 0
        self.needs_two_passes = True
        self.buffer = []
        self.fnames = {}
        self.fnames['s-test'] = []
        self.version = 5

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        # compute stuff from data
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_counts = cat_filt.spatial_counts()
            counts.append(gridded_counts)
        # we want to aggregate the counts in each bin to preserve memory
        if len(self.data) == 0:
            self.data = numpy.array(counts)
        else:
            self.data += numpy.array(counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the data
        time_horizon, n_cat, end_epoch, obs = args
        apprx_rate_density = numpy.array(self.data) / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1)

        # unfortunately, we need to iterate twice through the catalogs for this, unless we start pre-processing
        # everything and storing approximate cell-wise rates
        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_cat = cat_filt.spatial_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm
        self.test_distribution_spatial.append(lhs_norm)

    def post_process(self, obs, args=None):
        cata_iter, time_horizon, end_epoch, n_cat = args
        results = {}

        apprx_rate_density = numpy.array(self.data) / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1)

        # there can be nans in the spatial distribution
        test_distribution_spatial = numpy.array(self.test_distribution_spatial)
        # prepare results for each mw
        for i, mw in enumerate(self.mws):
            # get observed likelihood
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            if obs_filt.event_count == 0:
                print(f'Skipping pseudo-likelihood based tests for M{mw}+ because no events in observed observed_catalog.')
                continue
            n_obs = obs_filt.get_number_of_events()
            gridded_obs = obs_filt.spatial_counts()
            obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
            message = "normal"
            if obs_lh == -numpy.inf or obs_lh_norm == -numpy.inf:
                idx_good_sim = apprx_rate_density[i,:] != 0
                new_gridded_obs = gridded_obs[idx_good_sim]
                new_n_obs = numpy.sum(new_gridded_obs)
                print(f"Found -inf as the observed likelihood score for M{self.mws[i]}+. "
                      f"Assuming event(s) occurred in undersampled region of forecast.\n"
                      f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")
                if new_n_obs == 0:
                    print(f'Skipping pseudo-likelihood based tests for M{mw}+ because no events in observed observed_catalog '
                          f'after correcting for under-sampling in forecast.')
                    continue
                new_ard = apprx_rate_density[i,idx_good_sim]
                # we need to use the old n_obs here, because if we normalize the ard to a different value the observed
                # statistic will not be computed correctly.
                obs_lh, obs_lh_norm = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count[i], n_obs)
                message = "undersampled"

            # check for nans here
            test_distribution_spatial_1d = test_distribution_spatial[:,i]
            if numpy.isnan(numpy.sum(test_distribution_spatial_1d)):
                test_distribution_spatial_1d = test_distribution_spatial_1d[~numpy.isnan(test_distribution_spatial_1d)]

            if n_obs == 0 or numpy.isnan(obs_lh_norm):
                message = "not-valid"
                quantile_spatial = -1
            else:
                _, quantile_spatial = get_quantiles(test_distribution_spatial_1d, obs_lh_norm)

            result_spatial = EvaluationResult(test_distribution=test_distribution_spatial_1d,
                                          name='S-Test',
                                          observed_statistic=obs_lh_norm,
                                          quantile=quantile_spatial,
                                          status=message,
                                          min_mw=mw,
                                          obs_catalog_repr=obs.date_accessed,
                                          sim_name=self.name,
                                          obs_name=obs.name)

            results[mw] = result_spatial

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result in results.items():

            if result.status == 'not-valid':
                print(f'Skipping plot for spatial test on {mw}. Test results are not valid, likely because no earthquakes observed in target observed_catalog.')
                continue

            # plot spatial test
            s_test_fname = self._build_filename(plot_dir, mw, 's-test')
            plot_args = {'percentile': 95,
                         'title': f'Spatial Test, M{mw}+',
                         'filename': s_test_fname}
            _ = plot_spatial_test(result, axes=None, plot_args=plot_args, show=False)
            self.fnames['s-test'].append(s_test_fname)


class LikelihoodTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region = None
        self.test_distribution_likelihood = []
        self.cat_id = 0
        self.needs_two_passes = True
        self.buffer = []
        self.fnames = {}
        self.fnames['l-test'] = []
        self.fnames['s-test'] = []
        self.version = 5

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        # compute stuff from data
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_counts = cat_filt.spatial_counts()
            counts.append(gridded_counts)
        # we want to aggregate the counts in each bin to preserve memory
        if len(self.data) == 0:
            self.data = numpy.array(counts)
        else:
            self.data += numpy.array(counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the data
        time_horizon, n_cat, end_epoch, obs = args
        apprx_rate_density = numpy.array(self.data) / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1)

        # unfortunately, we need to iterate twice through the catalogs for this, unless we start pre-processing
        # everything and storing approximate cell-wise rates
        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))
        for i, mw in enumerate(self.mws):
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_cat = cat_filt.spatial_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm
        self.test_distribution_likelihood.append(lhs)

    def post_process(self, obs, args=None):
        cata_iter, time_horizon, end_epoch, n_cat = args
        results = {}

        apprx_rate_density = numpy.array(self.data) / n_cat
        expected_cond_count = numpy.sum(apprx_rate_density, axis=1)

        test_distribution_likelihood = numpy.array(self.test_distribution_likelihood)
        # there can be nans in the spatial distribution
        # prepare results for each mw
        for i, mw in enumerate(self.mws):
            # get observed likelihood
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            if obs_filt.event_count == 0:
                print(f'Skipping pseudo-likelihood based tests for M{mw}+ because no events in observed observed_catalog.')
                continue
            n_obs = obs_filt.get_number_of_events()
            gridded_obs = obs_filt.spatial_counts()
            obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density[i,:], expected_cond_count[i], n_obs)
            # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
            message = "normal"
            if obs_lh == -numpy.inf or obs_lh_norm == -numpy.inf:
                idx_good_sim = apprx_rate_density[i,:] != 0
                new_gridded_obs = gridded_obs[idx_good_sim]
                new_n_obs = numpy.sum(new_gridded_obs)
                print(f"Found -inf as the observed likelihood score for M{self.mws[i]}+. "
                      f"Assuming event(s) occurred in undersampled region of forecast.\n"
                      f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")
                if new_n_obs == 0:
                    print(f'Skipping pseudo-likelihood based tests for M{mw}+ because no events in observed observed_catalog '
                          f'after correcting for under-sampling in forecast.')
                    continue
                new_ard = apprx_rate_density[i,idx_good_sim]
                # we need to use the old n_obs here, because if we normalize the ard to a different value the observed
                # statistic will not be computed correctly.
                obs_lh, obs_lh_norm = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count[i], n_obs)
                message = "undersampled"

            # determine outcome of evaluation, check for infinity
            _, quantile_likelihood = get_quantiles(test_distribution_likelihood[:,i], obs_lh)

            # build evaluation result
            result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood[:,i],
                                                 name='L-Test',
                                                 observed_statistic=obs_lh,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 min_mw=mw,
                                                 obs_catalog_repr=obs.date_accessed,
                                                 sim_name=self.name,
                                                 obs_name=obs.name)

            results[mw] = result_likelihood

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result in results.items():
            # plot likelihood test
            l_test_fname = self._build_filename(plot_dir, mw, 'l-test')
            plot_args = {'percentile': 95,
                         'title': f'Pseudo-Likelihood Test, M{mw}+',
                         'filename': l_test_fname}
            _ = plot_likelihood_test(result, axes=None, plot_args=plot_args, show=show)

            # we can access this in the main program if needed
            # self.ax.append((ax, spatial_ax))
            self.fnames['s-test'].append(l_test_fname)


class CumulativeEventPlot(AbstractProcessingTask):

    def __init__(self, origin_epoch, end_epoch, **kwargs):
        super().__init__(**kwargs)
        self.origin_epoch = origin_epoch
        self.end_epoch = end_epoch
        self.time_bins, self.dt = self._get_time_bins()
        self.n_bins = self.time_bins.shape[0]
        self.archive = False

    def _get_time_bins(self):
        diff = (self.end_epoch - self.origin_epoch) / SECONDS_PER_DAY / 1000
        # if less than 7 day use hours
        if diff <= 7.0:
            dt = SECONDS_PER_HOUR * 1000
        # if less than 180 day use days
        elif diff <= 180:
            dt = SECONDS_PER_DAY * 1000
        # if less than 3 years (1,095.75 days) use weeks
        elif diff <= 1095.75:
            dt = SECONDS_PER_WEEK * 1000
        # use 30 day
        else:
            dt = SECONDS_PER_DAY * 1000 * 30
        # always make bins from start to end of observed_catalog
        return numpy.arange(self.origin_epoch, self.end_epoch+dt/2, dt), dt

    def process(self, catalog):
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            n_events = cat_filt.catalog.shape[0]
            ses_origin_time = cat_filt.get_epoch_times()
            inds = bin1d_vec(ses_origin_time, self.time_bins)
            binned_counts = numpy.zeros(self.n_bins)
            for j in range(n_events):
                binned_counts[inds[j]] += 1
            counts.append(binned_counts)
        self.data.append(counts)

    def post_process(self, obs, args=None):
        # data are stored as (n_cat, n_mw_bins, n_time_bins)
        summed_counts = numpy.cumsum(self.data, axis=2)
        # compute summary statistics for plotting
        fifth_per = numpy.percentile(summed_counts, 5, axis=0)
        first_quar = numpy.percentile(summed_counts, 25, axis=0)
        med_counts = numpy.percentile(summed_counts, 50, axis=0)
        second_quar = numpy.percentile(summed_counts, 75, axis=0)
        nine_fifth = numpy.percentile(summed_counts, 95, axis=0)
        # compute median for comcat observed_catalog
        obs_counts = []
        for mw in self.mws:
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            obs_binned_counts = numpy.zeros(self.n_bins)
            inds = bin1d_vec(obs_filt.get_epoch_times(), self.time_bins)
            for j in range(obs_filt.event_count):
                obs_binned_counts[inds[j]] += 1
            obs_counts.append(obs_binned_counts)
        obs_summed_counts = numpy.cumsum(obs_counts, axis=1)
        # update time_bins for plotting
        millis_to_hours = 60 * 60 * 1000 * 24
        time_bins = (self.time_bins - self.time_bins[0]) / millis_to_hours
        # since we are cumulating, plot at bin ends
        time_bins = time_bins + (self.dt / millis_to_hours)
        # make all arrays start at zero
        time_bins = numpy.insert(time_bins, 0, 0)
        # 2d array with (n_mw, n_time_bins)
        fifth_per = numpy.insert(fifth_per, 0, 0, axis=1)
        first_quar = numpy.insert(first_quar, 0, 0, axis=1)
        med_counts = numpy.insert(med_counts, 0, 0, axis=1)
        second_quar = numpy.insert(second_quar, 0, 0, axis=1)
        nine_fifth = numpy.insert(nine_fifth, 0, 0, axis=1)
        obs_summed_counts = numpy.insert(obs_summed_counts, 0, 0, axis=1)
        # ydata is now (5, n_mw, n_time_bins)
        results = {'xdata': time_bins,
                   'ydata': (fifth_per, first_quar, med_counts, second_quar, nine_fifth),
                   'obs_data': obs_summed_counts}

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # these are numpy arrays with mw information
        xdata = results['xdata']
        ydata = numpy.array(results['ydata'])
        obs_data = results['obs_data']
        # get values from plotting args
        for i, mw in enumerate(self.mws):
            cum_counts_fname = self._build_filename(plot_dir, mw, 'cum_counts')
            plot_args = {'title': f'Cumulative Event Counts, M{mw}+',
                         'xlabel': 'Days since start of forecast',
                         'filename': cum_counts_fname}
            ax = plot_cumulative_events_versus_time_dev(xdata, ydata[:,i,:], obs_data[i,:], plot_args, show=False)
            # self.ax.append(ax)
            self.fnames.append(cum_counts_fname)

    def store_results(self, results, dir):
        # store quickly for numpy, because we dont have a results class to deal with this
        fname = self._build_filename(dir, self.mws[0], 'cum_counts') + '.npy'
        numpy.save(fname, results)


class MagnitudeHistogram(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc = calc
        self.archive = False

    def process(self, catalog):
        """ this can share data with the Magnitude test, hence self.calc
        """
        if not self.name:
            self.name = catalog.name
        if self.calc:
            # always compute this for the lowest magnitude, above this is redundant
            cat_filt = catalog.filter(f'magnitude >= {self.mws[0]}')
            binned_mags = cat_filt.magnitude_counts()
            self.data.append(binned_mags)

    def post_process(self, obs, args=None):
        """ just store observation for later """
        _ = args
        self.obs = obs

    def plot(self, results, plot_dir, plot_args=None, show=False):
        mag_hist_fname = self._build_filename(plot_dir, self.mws[0], 'mag_hist')
        plot_args = {
             'xlim': [self.mws[0], numpy.max(CSEP_MW_BINS)],
             'title': f"Magnitude Histogram, M{self.mws[0]}+",
             'sim_label': self.name,
             'obs_label': self.obs.name,
             'filename': mag_hist_fname
        }
        obs_filt = self.obs.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        # data (n_sim, n_mag, n_mw_bins)
        ax = plot_magnitude_histogram_dev(numpy.array(self.data)[:,0,:], obs_filt, plot_args, show=False)
        # self.ax.append(ax)
        self.fnames.append(mag_hist_fname)


class UniformLikelihoodCalculation(AbstractProcessingTask):
    """
    This calculation assumes that the spatial distribution of the forecast is uniform, but the seismicity is located
    in spatial bins according to the clustering provided by the forecast model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.test_distribution_likelihood = []
        self.test_distribution_spatial = []
        self.fnames = {}
        self.fnames['l-test'] = []
        self.fnames['s-test'] = []
        self.needs_two_passes = True

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name

    def process_again(self, catalog, args=()):

        time_horizon, n_cat, end_epoch, obs = args

        expected_cond_count = numpy.sum(self.data, axis=1) / n_cat

        lhs = numpy.zeros(len(self.mws))
        lhs_norm = numpy.zeros(len(self.mws))

        for i, mw in enumerate(self.mws):

            # generate with uniform rate in every spatial bin
            apprx_rate_density = expected_cond_count[i] * numpy.ones(self.region.num_nodes) / self.region.num_nodes

            # convert to rate density
            apprx_rate_density = apprx_rate_density / self.region.dh / self.region.dh / time_horizon

            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            n_obs = obs_filt.event_count
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_cat = cat_filt.spatial_counts()
            lh, lh_norm = _compute_likelihood(gridded_cat, apprx_rate_density, expected_cond_count[i], n_obs)
            lhs[i] = lh
            lhs_norm[i] = lh_norm

        self.test_distribution_likelihood.append(lhs)
        self.test_distribution_spatial.append(lhs_norm)

    def post_process(self, obs, args=None):

        _, time_horizon, _, n_cat = args

        results = {}
        expected_cond_count = numpy.sum(self.data, axis=1) / n_cat

        test_distribution_likelihood = numpy.array(self.test_distribution_likelihood)
        test_distribution_spatial = numpy.array(self.test_distribution_spatial)

        for i, mw in enumerate(self.mws):

            # create uniform apprx rate density
            apprx_rate_density = expected_cond_count[i] * numpy.ones(self.region.num_nodes) / self.region.num_nodes

            # convert to rate density
            apprx_rate_density = apprx_rate_density / self.region.dh / self.region.dh / time_horizon

            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            n_obs = obs_filt.get_number_of_events()
            gridded_obs = obs_filt.spatial_counts()
            obs_lh, obs_lh_norm = _compute_likelihood(gridded_obs, apprx_rate_density, expected_cond_count[i],
                                                      n_obs)
            # determine outcome of evaluation, check for infinity
            _, quantile_likelihood = get_quantiles(test_distribution_likelihood[:, i], obs_lh)
            _, quantile_spatial = get_quantiles(test_distribution_spatial[:, i], obs_lh_norm)

            # Signals outcome of test
            message = "normal"
            # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
            # either normal and wrong or udetermined (undersampled)
            if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh):
                    # Build message
                    message = "undetermined"

            # build evaluation result
            result_likelihood = EvaluationResult(test_distribution=test_distribution_likelihood[:, i],
                                                 name='UL-Test',
                                                 observed_statistic=obs_lh,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 min_mw=mw,
                                                 obs_catalog_repr=obs.date_accessed,
                                                 sim_name=self.name,
                                                 obs_name=obs.name)
            # find out if there are issues with the test
            if numpy.isclose(quantile_spatial, 0.0) or numpy.isclose(quantile_spatial, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_lh_norm):
                    # Build message
                    message = "undetermined"

            if n_obs == 0:
                message = 'not-valid'

            result_spatial = EvaluationResult(test_distribution=test_distribution_spatial[:, i],
                                              name='US-Test',
                                              observed_statistic=obs_lh_norm,
                                              quantile=quantile_spatial,
                                              status=message,
                                              min_mw=mw,
                                              obs_catalog_repr=obs.date_accessed,
                                              sim_name=self.name,
                                              obs_name=obs.name)

            results[mw] = (result_likelihood, result_spatial)

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result_tuple in results.items():
            # plot likelihood test
            l_test_fname = self._build_filename(plot_dir, mw, 'ul-test')
            plot_args = {'percentile': 95,
                         'title': f'Pseudo-Likelihood Test\nMw > {mw}',
                         'bins': 'fd',
                         'filename': l_test_fname}
            _ = plot_likelihood_test(result_tuple[0], axes=None, plot_args=plot_args, show=show)

            # we can access this in the main program if needed
            # self.ax.append((ax, spatial_ax))
            self.fnames['l-test'].append(l_test_fname)

            if result_tuple[1].status == 'not-valid':
                print(
                    f'Skipping plot for spatial test on {mw}. Test results are not valid, likely because no earthquakes observed in target observed_catalog.')
                continue

            # plot spatial test
            s_test_fname = self._build_filename(plot_dir, mw, 'us-test')
            plot_args = {'percentile': 95,
                         'title': f'Spatial Test\nMw > {mw}',
                         'bins': 'fd',
                         'filename': s_test_fname}
            _ = plot_spatial_test(result_tuple[1], axes=None, plot_args=plot_args, show=False)
            self.fnames['s-test'].append(s_test_fname)


class InterEventTimeDistribution(AbstractProcessingTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mws = [2.5]
        # but this should be smart based on the length of the observed_catalog
        self.data = AdaptiveHistogram(dh=0.1)
        self.test_distribution = []
        self.needs_two_passes = True
        # jsut saves soem computation bc we only need to compute this once
        self.normed_data = numpy.array([])
        self.version = 2

    def process(self, catalog):
        if self.name is None:
            self.name = catalog.name
        cat_ietd = catalog.get_inter_event_times()
        self.data.add(cat_ietd)

    def process_again(self, catalog, args=()):
        cat_ietd = catalog.get_inter_event_times()
        disc_ietd = numpy.zeros(len(self.data.bins))
        idx = bin1d_vec(cat_ietd, self.data.bins)
        numpy.add.at(disc_ietd, idx, 1)
        disc_ietd_normed = numpy.cumsum(disc_ietd) / numpy.sum(disc_ietd)
        if self.normed_data.size == 0:
            self.normed_data = numpy.cumsum(self.data.data) / numpy.sum(self.data.data)
        self.test_distribution.append(sup_dist(self.normed_data, disc_ietd_normed))

    def post_process(self, obs, args=None):
        # get inter-event times from observed_catalog
        obs_filt = obs.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        obs_ietd = obs_filt.get_inter_event_times()
        obs_disc_ietd = numpy.zeros(len(self.data.bins))
        idx = bin1d_vec(obs_ietd, self.data.bins)
        numpy.add.at(obs_disc_ietd, idx, 1)
        obs_disc_ietd_normed = numpy.cumsum(obs_disc_ietd) / numpy.trapz(obs_disc_ietd)
        d_obs = sup_dist(self.normed_data, obs_disc_ietd_normed)
        _, quantile = get_quantiles(self.test_distribution, d_obs)
        result = EvaluationResult(test_distribution=self.test_distribution,
                                  name='IETD-Test',
                                  observed_statistic=d_obs,
                                  quantile=quantile,
                                  status='Normal',
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        ietd_test_fname = self._build_filename(plot_dir, results.min_mw, 'ietd_test')
        _ = plot_distribution_test(results, show=False, plot_args={'percentile': 95,
                                                                   'title': f'Inter-event Time Distribution Test, M{results.min_mw}+',
                                                                   'bins': 'auto',
                                                                   'xlabel': "D* Statistic",
                                                                   'ylabel': r"Number of catalogs",
                                                                   'filename': ietd_test_fname})
        self.fnames.append(ietd_test_fname)


class InterEventDistanceDistribution(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mws = [2.5]
        # start by using a 10 second bin for discretizing data
        # but this should be smart based on the length of the observed_catalog
        self.data = AdaptiveHistogram(dh=1)
        self.test_distribution = []
        self.needs_two_passes = True
        # jsut saves soem computation bc we only need to compute this once
        self.normed_data = numpy.array([])
        self.version = 2

    def process(self, catalog):
        """ not nice on the memorys. """
        if self.name is None:
            self.name = catalog.name
        # distances are in kilometers
        cat_iedd = catalog.get_inter_event_distances()
        self.data.add(cat_iedd)

    def process_again(self, catalog, args=()):
        cat_iedd = catalog.get_inter_event_distances()
        disc_iedd = numpy.zeros(len(self.data.bins))
        idx = bin1d_vec(cat_iedd, self.data.bins)
        numpy.add.at(disc_iedd, idx, 1)
        disc_iedd_normed = numpy.cumsum(disc_iedd) / numpy.sum(disc_iedd)
        if self.normed_data.size == 0:
            self.normed_data = numpy.cumsum(self.data.data) / numpy.sum(self.data.data)
        self.test_distribution.append(sup_dist(self.normed_data, disc_iedd_normed))

    def post_process(self, obs, args=None):
        # get inter-event times from data
        obs_filt = obs.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        obs_iedd = obs_filt.get_inter_event_distances()
        obs_disc_iedd = numpy.zeros(len(self.data.bins))
        idx = bin1d_vec(obs_iedd, self.data.bins)
        numpy.add.at(obs_disc_iedd, idx, 1)
        obs_disc_iedd_normed = numpy.cumsum(obs_disc_iedd) / numpy.trapz(obs_disc_iedd)
        d_obs = sup_dist(self.normed_data, obs_disc_iedd_normed)
        _, quantile = get_quantiles(self.test_distribution, d_obs)
        result = EvaluationResult(test_distribution=self.test_distribution,
                                  name='IEDD-Test',
                                  observed_statistic=d_obs,
                                  quantile=quantile,
                                  status='Normal',
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        iedd_test_fname = self._build_filename(plot_dir, results.min_mw, 'iedd_test')
        _ = plot_distribution_test(results, show=False, plot_args={'percentile': 95,
                                                                   'title': f'Inter-event Distance Distribution Test, M{results.min_mw}+',
                                                                   'bins': 'auto',
                                                                   'xlabel': "D* statistic",
                                                                   'ylabel': r"Number of catalogs",
                                                                   'filename': iedd_test_fname})
        self.fnames.append(iedd_test_fname)


class TotalEventRateDistribution(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.needs_two_passes = True
        self.data = AdaptiveHistogram(dh=1)
        self.normed_data = numpy.array([])
        self.test_distribution = []
        self.version = 2

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        # compute stuff from observed_catalog
        gridded_counts = catalog.spatial_counts()
        self.data.add(gridded_counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the data
        _, n_cat, _, _ = args
        cat_counts = catalog.spatial_counts()
        cat_disc = numpy.zeros(len(self.data.bins))
        idx = bin1d_vec(cat_counts, self.data.bins)
        numpy.add.at(cat_disc, idx, 1)
        disc_terd_normed = numpy.cumsum(cat_disc) / numpy.sum(cat_disc)
        if self.normed_data.size == 0:
            self.normed_data = numpy.cumsum(self.data.data) / numpy.sum(self.data.data)
        self.test_distribution.append(sup_dist(self.normed_data, disc_terd_normed))

    def post_process(self, obs, args=None):
        # get inter-event times from observed_catalog
        obs_filt = obs.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        obs_terd = obs_filt.spatial_counts()
        obs_disc_terd = numpy.zeros(len(self.data.bins))
        idx = bin1d_vec(obs_terd, self.data.bins)
        numpy.add.at(obs_disc_terd, idx, 1)
        obs_disc_terd_normed = numpy.cumsum(obs_disc_terd) / numpy.sum(obs_disc_terd)
        d_obs = sup_dist(self.normed_data, obs_disc_terd_normed)
        _, quantile = get_quantiles(self.test_distribution, d_obs)
        result = EvaluationResult(test_distribution=self.test_distribution,
                                  name='TERD-Test',
                                  observed_statistic=d_obs,
                                  quantile=quantile,
                                  status='Normal',
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)

        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        terd_test_fname = AbstractProcessingTask._build_filename(plot_dir, results.min_mw, 'terd_test')
        _ = plot_distribution_test(results, show=False, plot_args={'percentile': 95,
                                                                  'title': f'Total Event Rate Distribution-Test, M{results.min_mw}+',
                                                                  'bins': 'auto',
                                                                  'xlabel': "D* Statistic",
                                                                  'ylabel': r"Number of catalogs",
                                                                  'filename': terd_test_fname})
        self.fnames.append(terd_test_fname)


class BValueTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.version = 2

    def process(self, catalog):
        if not self.name:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        self.data.append(cat_filt.get_bvalue(reterr=False))

    def post_process(self, obs, args=None):
        _ = args
        data = numpy.array(self.data)
        obs_filt = obs.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        obs_bval = obs_filt.get_bvalue(reterr=False)
        # get delta_1 and delta_2 values
        _, delta_2 = get_quantiles(data, obs_bval)
        # prepare result
        result = EvaluationResult(test_distribution=data,
                                  name='BV-Test',
                                  observed_statistic=obs_bval,
                                  quantile=delta_2,
                                  status='Normal',
                                  min_mw=self.mws[0],
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)
        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        bv_test_fname = self._build_filename(plot_dir, results.min_mw, 'bv_test')
        _ = plot_number_test(results, show=False, plot_args={'percentile': 95,
                                                             'title': f"B-Value Distribution Test, M{results.min_mw}+",
                                                             'bins': 'auto',
                                                             'xlabel': 'b-value',
                                                             'xy': (0.2, 0.65),
                                                             'filename': bv_test_fname})
        self.fnames.append(bv_test_fname)


class MedianMagnitudeTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, catalog):
        if not self.name:
            self.name = catalog.name
        cat_filt = catalog.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        self.data.append(numpy.median(cat_filt.get_magnitudes()))

    def post_process(self, obs, args=None):
        _ = args
        data = numpy.array(self.data)
        obs_filt = obs.filter(f'magnitude >= {self.mws[0]}', in_place=False)
        observation_count = float(numpy.median(obs_filt.get_magnitudes()))
        # get delta_1 and delta_2 values
        _, delta_2 = get_quantiles(data, observation_count)
        # prepare result
        result = EvaluationResult(test_distribution=data,
                                  name='M-Test',
                                  observed_statistic=observation_count,
                                  quantile=delta_2,
                                  min_mw=self.mws[0],
                                  status='Normal',
                                  obs_catalog_repr=obs.date_accessed,
                                  sim_name=self.name,
                                  obs_name=obs.name)
        return result

    def plot(self, results, plot_dir, plot_args=None, show=False):
        mm_test_fname = self._build_filename(plot_dir, self.mws[0], 'mm_test')
        _ = plot_number_test(results, show=False, plot_args={'percentile': 95,
                                                             'title': f"Median Magnitude Distribution Test\nMw > {self.mws[0]}",
                                                             'bins': 25,
                                                             'filename': mm_test_fname})
        self.fnames.append(mm_test_fname)


class SpatialProbabilityTest(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region = None
        self.test_distribution = []
        self.needs_two_passes = True
        self.buffer = []
        self.fnames = []
        self.version = 3

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        # compute stuff from data
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_counts = cat_filt.spatial_event_probability()
            counts.append(gridded_counts)
        # we want to aggregate the counts in each bin to preserve memory
        if len(self.data) == 0:
            self.data = numpy.array(counts)
        else:
            self.data += numpy.array(counts)

    def process_again(self, catalog, args=()):
        # we dont actually need to do this if we are caching the data
        time_horizon, n_cat, end_epoch, obs = args
        with numpy.errstate(divide='ignore'):
            prob_map = numpy.log10(self.data / n_cat)

        # unfortunately, we need to iterate twice through the catalogs for this.
        probs = numpy.zeros(len(self.mws))
        for i, mw in enumerate(self.mws):
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            gridded_cat = cat_filt.spatial_event_probability()
            prob = _compute_spatial_statistic(gridded_cat, prob_map[i, :])
            probs[i] = prob
        self.test_distribution.append(probs)

    def post_process(self, obs, args=None):
        cata_iter, time_horizon, end_epoch, n_cat = args
        results = {}
        with numpy.errstate(divide='ignore'):
            prob_map = numpy.log10(self.data / n_cat)
        test_distribution_prob = numpy.array(self.test_distribution)
        # prepare results for each mw
        for i, mw in enumerate(self.mws):
            # get observed likelihood
            obs_filt = obs.filter(f'magnitude >= {mw}', in_place=False)
            if obs_filt.event_count == 0:
                print(f'Skipping Probability test for Mw {mw} because no events in observed observed_catalog.')
                continue
            gridded_obs = obs_filt.spatial_event_probability()
            obs_prob = _compute_spatial_statistic(gridded_obs, prob_map[i, :])
            # determine outcome of evaluation, check for infinity will go here...
            test_1d = test_distribution_prob[:,i]
            if numpy.isnan(numpy.sum(test_1d)):
                test_1d = test_1d[~numpy.isnan(test_1d)]
            _, quantile_likelihood = get_quantiles(test_1d, obs_prob)
            # Signals outcome of test
            message = "normal"
            # Deal with case with cond. rate. density func has zeros. Keep value but flag as being
            # either normal and wrong or udetermined (undersampled)
            if numpy.isclose(quantile_likelihood, 0.0) or numpy.isclose(quantile_likelihood, 1.0):
                # undetermined failure of the test
                if numpy.isinf(obs_prob):
                    # Build message, should maybe try sampling procedure from pseudo-likelihood based tests
                    message = "undetermined"
            # build evaluation result
            result_prob = EvaluationResult(test_distribution=test_1d,
                                                 name='Prob-Test',
                                                 observed_statistic=obs_prob,
                                                 quantile=quantile_likelihood,
                                                 status=message,
                                                 min_mw=mw,
                                                 obs_catalog_repr=obs.date_accessed,
                                                 sim_name=self.name,
                                                 obs_name=obs.name)
            results[mw] = result_prob

        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw, result in results.items():
            # plot likelihood test
            prob_test_fname = self._build_filename(plot_dir, mw, 'prob-test')
            plot_args = {'percentile': 95,
                         'title': f'Probability Test, M{mw}+',
                         'bins': 'auto',
                         'xlabel': 'Spatial probability statistic',
                         'ylabel': 'Number of catalogs',
                         'filename': prob_test_fname}
            _ = plot_probability_test(result, axes=None, plot_args=plot_args, show=show)
            self.fnames.append(prob_test_fname)


class SpatialProbabilityPlot(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc=calc
        self.region=None
        self.archive=False

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        if self.calc:
            # compute stuff from data
            counts = []
            for mw in self.mws:
                cat_filt = catalog.filter(f'magnitude >= {mw}')
                gridded_counts = cat_filt.spatial_event_probability()
                counts.append(gridded_counts)
            # we want to aggregate the counts in each bin to preserve memory
            if len(self.data) == 0:
                self.data = numpy.array(counts)
            else:
                self.data += numpy.array(counts)

    def post_process(self, obs, args=None):
        """ store things for later """
        self.obs = obs
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        return None

    def plot(self, results, plot_dir, plot_args=None, show=False):
        with numpy.errstate(divide='ignore'):
            prob = numpy.log10(numpy.array(self.data) / self.n_cat)
        for i, mw in enumerate(self.mws):
            # compute expected rate density
            obs_filt = self.obs.filter(f'magnitude >= {mw}', in_place=False)
            plot_data = self.region.get_cartesian(prob[i,:])
            ax = plot_spatial_dataset(plot_data,
                                      self.region,
                                      plot_args={'clabel': r'Log$_{10}$ Probability 1 or more events'
                                                           '\n'
                                                           f'within {self.region.dh}°x{self.region.dh}° cells',
                                                 'clim': [-5, 0],
                                                 'title': f'Spatial Probability Plot, M{mw}+'})
            ax.scatter(obs_filt.get_longitudes(), obs_filt.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
            crd_fname = self._build_filename(plot_dir, mw, 'prob_obs')
            ax.figure.savefig(crd_fname + '.png')
            ax.figure.savefig(crd_fname + '.pdf')
            self.fnames.append(crd_fname)


class ApproximateRatePlot(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc=calc
        self.region=None
        self.archive = False
        self.version = 2

    def process(self, data):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = data.region
        if not self.name:
            self.name = data.name
        if self.calc:
            # compute stuff from data
            counts = []
            for mw in self.mws:
                cat_filt = data.filter(f'magnitude >= {mw}')
                gridded_counts = cat_filt.spatial_counts()
                counts.append(gridded_counts)
            # we want to aggregate the counts in each bin to preserve memory
            if len(self.data) == 0:
                self.data = numpy.array(counts)
            else:
                self.data += numpy.array(counts)

    def post_process(self, obs, args=None):
        """ store things for later """
        self.obs = obs
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        return None

    def plot(self, results, plot_dir, plot_args=None, show=False):
        with numpy.errstate(divide='ignore'):
            crd = numpy.log10(numpy.array(self.data) / self.n_cat)

        for i, mw in enumerate(self.mws):
            # compute expected rate density
            obs_filt = self.obs.filter(f'magnitude >= {mw}', in_place=False)
            plot_data = self.region.get_cartesian(crd[i,:])
            ax = plot_spatial_dataset(plot_data,
                                      self.region,
                                      plot_args={'clabel': r'Log$_{10}$ Approximate rate density'
                                                           '\n'
                                                           f'(Expected events per week per {self.region.dh}°x{self.region.dh}°)',
                                                 'clim': [-5, 0],
                                                 'title': f'Approximate Rate Density with Observations, M{mw}+'})
            ax.scatter(obs_filt.get_longitudes(), obs_filt.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
            crd_fname = self._build_filename(plot_dir, mw, 'crd_obs')
            ax.figure.savefig(crd_fname + '.png')
            ax.figure.savefig(crd_fname + '.pdf')
            # self.ax.append(ax)
            self.fnames.append(crd_fname)


class ApproximateRateDensity(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc = calc
        self.region = None
        self.archive = False
        self.mag_dh = None

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        if not self.mag_dh:
            mag_dh = self.region.magnitudes[1] - self.region.magnitudes[0]
            self.mag_dh = mag_dh
        if self.calc:
            # compute stuff from data
            gridded_counts = catalog.spatial_magnitude_counts()
            # we want to aggregate the counts in each bin to preserve memory
            if self.n_cat is not None:
                if len(self.data) == 0:
                    self.data = numpy.array(gridded_counts) / self.n_cat
                else:
                    self.data += numpy.array(gridded_counts) / self.n_cat
            else:
                if len(self.data) == 0:
                    self.data = numpy.array(gridded_counts)
                else:
                    self.data += numpy.array(gridded_counts)

    def post_process(self, obs, args=()):
        """ store things for later, and call if n_cat was not availabe at run-time for some reason. """
        self.obs = obs
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        with numpy.errstate(divide='ignore'):
            self.crd = numpy.array(self.data) / self.n_cat
        return None

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # compute expected rate density
        with numpy.errstate(divide='ignore'):
            plot_data = numpy.log10(self.region.get_cartesian(self.crd))
        ax = plot_spatial_dataset(plot_data,
                                  self.region,
                                  plot_args={'clabel': r'Log$_{10}$ Approximate Rate Density'
                                                       '\n'
                                                       f'(Expected Events per year per {self.region.dh}°x{self.region.dh}°) per {self.mag_dh} Mw',
                                             'clim': [0, 5],
                                             'title': f'Approximate Rate Density with Observations, M{self.min_mw}+'})
        ax.scatter(self.obs.get_longitudes(), self.obs.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
        crd_fname = self._build_filename(plot_dir, self.min_mw, 'crd_obs')
        ax.figure.savefig(crd_fname + '.png')
        ax.figure.savefig(crd_fname + '.pdf')
        # self.ax.append(ax)
        self.fnames.append(crd_fname)


class ApproximateSpatialRateDensity(AbstractProcessingTask):
    def __init__(self, calc=True, **kwargs):
        super().__init__(**kwargs)
        self.calc = calc
        self.region = None
        self.archive = False

    def process(self, catalog):
        # grab stuff from data that we might need later
        if not self.region:
            self.region = catalog.region
        if not self.name:
            self.name = catalog.name
        if self.calc:
            # compute stuff from data
            gridded_counts = catalog.spatial_counts()
            # we want to aggregate the counts in each bin to preserve memory
            if len(self.data) == 0:
                self.data = numpy.array(gridded_counts)
            else:
                self.data += numpy.array(gridded_counts)

    def post_process(self, obs, args=()):
        """ store things for later """
        self.obs = obs
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        self.crd = numpy.array(self.data) / self.region.dh / self.region.dh / self.time_horizon / self.n_cat
        return None

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # compute expected rate density
        with numpy.errstate(divide='ignore'):
            plot_data = numpy.log10(self.region.get_cartesian(self.crd))
        ax = plot_spatial_dataset(plot_data,
                                  self.region,
                                  plot_args={'clabel': r'Log$_{10}$ Approximate Rate Density'
                                                       '\n'
                                                       f'(Expected Events per year per {self.region.dh}°x{self.region.dh}°)',
                                             'clim': [0, 5],
                                             'title': f'Approximate Rate Density with Observations, M{self.min_mw}+'})
        ax.scatter(self.obs.get_longitudes(), self.obs.get_latitudes(), marker='.', color='white', s=40, edgecolors='black')
        crd_fname = self._build_filename(plot_dir, self.min_mw, 'crd_obs')
        ax.figure.savefig(crd_fname + '.png')
        ax.figure.savefig(crd_fname + '.pdf')
        # self.ax.append(ax)
        self.fnames.append(crd_fname)


class ConditionalApproximateRatePlot(AbstractProcessingTask):
    def __init__(self, obs, **kwargs):
        super().__init__(**kwargs)
        self.obs = obs
        self.data = defaultdict(list)
        self.archive = False
        self.version = 2

    def process(self, data):
        if self.name is None:
            self.name = data.name

        if self.region is None:
            self.region = data.region
        """ collects all catalogs conforming to n_obs in a dict"""
        for mw in self.mws:
            cat_filt = data.filter(f'magnitude >= {mw}')
            obs_filt = self.obs.filter(f'magnitude >= {mw}', in_place=False)
            n_obs = obs_filt.event_count
            tolerance = 0.05 * n_obs
            if cat_filt.event_count <= n_obs + tolerance \
                and cat_filt.event_count >= n_obs - tolerance:
                self.data[mw].append(cat_filt.spatial_counts())

    def post_process(self, obs, args=None):
        _, time_horizon, _, n_cat = args
        self.time_horizon = time_horizon
        self.n_cat = n_cat
        return

    def plot(self, results, plot_dir, plot_args=None, show=False):
        # compute conditional approximate rate density
        for i, mw in enumerate(self.mws):
            # compute expected rate density
            obs_filt = self.obs.filter(f'magnitude >= {mw}', in_place=False)
            if obs_filt.event_count == 0:
                continue

            rates = numpy.array(self.data[mw])
            if rates.shape[0] == 0:
                continue

            # compute conditional approximate rate
            mean_rates = numpy.mean(rates, axis=0)
            with numpy.errstate(divide='ignore'):
                crd = numpy.log10(mean_rates)
            plot_data = self.region.get_cartesian(crd)
            ax = plot_spatial_dataset(plot_data,
                                      self.region,
                                      plot_args={'clabel': r'Log$_{10}$ Conditional Rate Density'
                                                           '\n'
                                      f'(Expected Events per year per {self.region.dh}°x{self.region.dh}°)',
                                                 'clim': [-5, 0],
                                                 'title': f'Conditional Approximate Rate Density with Observations, M{mw}+'})
            ax.scatter(obs_filt.get_longitudes(), obs_filt.get_latitudes(), marker='.', color='white', s=40,
                       edgecolors='black')
            crd_fname = self._build_filename(plot_dir, mw, 'cond_rates')
            ax.figure.savefig(crd_fname + '.png')
            ax.figure.savefig(crd_fname + '.pdf')
                # self.ax.append(ax)
            self.fnames.append(crd_fname)


class CatalogMeanStabilityAnalysis(AbstractProcessingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calc = False
        self.mws = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

    def process(self, catalog):
        if not self.name:
            self.name = catalog.name
        counts = []
        for mw in self.mws:
            cat_filt = catalog.filter(f'magnitude >= {mw}')
            counts.append(cat_filt.event_count)
        self.data.append(counts)

    def post_process(self, obs, args=None):
        results = {}
        data = numpy.array(self.data)
        n_sim = data.shape[0]
        end_points = numpy.arange(1,n_sim,100)
        for i, mw in enumerate(self.mws):
            running_means = []
            for ep in end_points:
                running_means.append(numpy.mean(data[:ep,i]))
            results[mw] = (end_points, running_means)
        return results

    def plot(self, results, plot_dir, plot_args=None, show=False):
        for mw in self.mws:
            fname = self._build_filename(plot_dir, mw, 'comp_test')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            res = numpy.array(results[mw])
            ax.plot(res[0,:], res[1,:])
            ax.set_title(f'Catalog Mean Stability Mw > {mw}')
            ax.set_xlabel('Average Event Count')
            ax.set_ylabel('Running Mean')
            fig.savefig(fname + '.png')
            plt.close(fig)