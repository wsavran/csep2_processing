
import os
import json
import copy
import time

import numpy

import datetime
import matplotlib

import seaborn as sns

from csep import load_stochastic_event_sets, query_comcat
from csep.utils import current_git_hash
from csep.utils.time_utils import epoch_time_to_utc_datetime, datetime_to_utc_epoch, millis_to_days, utc_now_epoch
from csep.core.processing_tasks import NumberTest, MagnitudeTest, LikelihoodAndSpatialTest, CumulativeEventPlot, \
    MagnitudeHistogram, InterEventTimeDistribution, InterEventDistanceDistribution, TotalEventRateDistribution, \
    BValueTest, SpatialProbabilityTest, SpatialProbabilityPlot, ApproximateRatePlot, ConditionalApproximateRatePlot
from csep.core.regions import california_relm_region, masked_region
from csep.utils.basic_types import Polygon
from csep.utils.scaling_relationships import WellsAndCoppersmith
from csep.utils.comcat import get_event_by_id
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR
from csep.utils.file import get_relative_path, mkdirs, copy_file
from csep.utils.documents import MarkdownReport
from csep.models import EvaluationConfiguration, Event
from csep.core.catalogs import ComcatCatalog
from csep.core.repositories import FileSystem

def ucerf3_consistency_testing(sim_dir, event_id, end_epoch, n_cat=None, plot_dir=None, generate_markdown=True, catalog_repo=None, save_results=False,
                               force_plot_all=False, skip_processing=False, event_repo=None, name=''):
    """
    computes all csep consistency tests for simulation located in sim_dir with event_id

    Args:
        sim_dir (str): directory where results and configuration are stored
        event_id (str): event_id corresponding to comcat event
    """
    # set up directories
    matplotlib.use('agg')
    matplotlib.rcParams['figure.max_open_warning'] = 150
    sns.set()

    # try using two different files
    print(f"Processing simulation in {sim_dir}", flush=True)
    filename = os.path.join(sim_dir, 'results_complete.bin')
    if not os.path.exists(filename):
        filename = os.path.join(sim_dir, 'results_complete_partial.bin')
    if not os.path.exists(filename):
        raise FileNotFoundError('could not find results_complete.bin or results_complete_partial.bin')
        
    if plot_dir is None:
        plot_dir = sim_dir
        print(f'No plotting directory specified defaulting to {plot_dir}')
    else:
        print(f"Using user specified plotting directory: {plot_dir}")

    # config file can be either config.json or basename of simulation-config.json
    config_file = os.path.join(sim_dir, 'config.json')
    if not os.path.exists(config_file):
        config_file = os.path.join(sim_dir, os.path.basename(sim_dir) + '-config.json')
    mkdirs(os.path.join(plot_dir))

    # observed_catalog filename
    catalog_fname = os.path.join(plot_dir, 'evaluation_catalog.json')

    # load ucerf3 configuration
    with open(os.path.join(config_file), 'r') as f:
        u3etas_config = json.load(f)

    if plot_dir != sim_dir:
        print("Plotting dir is different than simulation directory. copying simulation configuration to plot directory")
        copy_file(config_file, os.path.join(plot_dir, 'config.json'))

    # determine how many catalogs to process
    if n_cat is None or n_cat > u3etas_config['numSimulations']:
        n_cat = u3etas_config['numSimulations']

    # download comcat information, sometimes times out but usually doesn't fail twice in a row
    if event_repo is not None:
        print("Using event information stored instead of accessing ComCat.")
        event_repo = FileSystem(url=event_repo)
        event = event_repo.load(Event())
    else:
        try:
            event = get_event_by_id(event_id)
        except:
            event = get_event_by_id(event_id)

    # filter to aftershock radius
    rupture_length = WellsAndCoppersmith.mag_length_strike_slip(event.magnitude) * 1000
    aftershock_polygon = Polygon.from_great_circle_radius((event.longitude, event.latitude),
                                                          3*rupture_length, num_points=100)
    aftershock_region = masked_region(california_relm_region(dh_scale=4), aftershock_polygon)

    # event timing
    event_time = event.time.replace(tzinfo=datetime.timezone.utc)
    event_epoch = datetime_to_utc_epoch(event.time)
    origin_epoch = u3etas_config['startTimeMillis']

    # this kinda booty, should probably add another variable or something
    if type(end_epoch) == str:
        print(f'Found end_epoch as time_delta string (in days), adding {end_epoch} days to simulation start time')
        time_delta = 1000*24*60*60*int(end_epoch)
        end_epoch = origin_epoch + time_delta

    # convert epoch time (millis) to years
    time_horizon = (end_epoch - origin_epoch) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000

    # Download comcat observed_catalog, if it fails its usually means it timed out, so just try again
    if catalog_repo is None:
        print("Catalog repository not specified downloading new observed_catalog from ComCat.")

        # Sometimes ComCat fails for non-critical reasons, try twice just to make sure.
        try:
            comcat = query_comcat(epoch_time_to_utc_datetime(origin_epoch), epoch_time_to_utc_datetime(end_epoch),
                                  min_magnitude=2.50,
                                  min_latitude=31.50, max_latitude=43.00,
                                  min_longitude=-125.40, max_longitude=-113.10)
            comcat = comcat.filter_spatial(aftershock_region).apply_mct(event.magnitude, event_epoch)
            print(comcat)
        except:
            comcat = query_comcat(event_time, epoch_time_to_utc_datetime(end_epoch),
                                  min_magnitude=2.50,
                                  min_latitude=31.50, max_latitude=43.00,
                                  min_longitude=-125.40, max_longitude=-113.10)
            comcat = comcat.filter_spatial(aftershock_region).apply_mct(event.magnitude, event_epoch)
            print(comcat)
    else:
        # if this fails it should stop the program, therefore no try-catch block
        print(f"Reading observed_catalog from repository at location {catalog_repo}")
        catalog_repo = FileSystem(url=catalog_repo)
        comcat = catalog_repo.load(ComcatCatalog(query=False))
        comcat = comcat.filter(f'origin_time >= {origin_epoch}').filter(f'origin_time < {end_epoch}')
        comcat = comcat.filter_spatial(aftershock_region).apply_mct(event.magnitude, event_epoch)
        print(comcat)

    # define products to compute on simulation, this could be extracted
    data_products = {
         'n-test': NumberTest(),
         'm-test': MagnitudeTest(),
         'l-test': LikelihoodAndSpatialTest(),
         'cum-plot': CumulativeEventPlot(origin_epoch, end_epoch),
         'mag-hist': MagnitudeHistogram(),
         'arp-plot': ApproximateRatePlot(),
         'prob-plot': SpatialProbabilityPlot(),
         'prob-test': SpatialProbabilityTest(),
         'carp-plot': ConditionalApproximateRatePlot(comcat),
         'terd-test': TotalEventRateDistribution(),
         'iedd-test': InterEventDistanceDistribution(),
         'ietd-test': InterEventTimeDistribution(),
         'bv-test': BValueTest()
    }

    # try and read metadata file from plotting dir
    metadata_fname = os.path.join(plot_dir, 'meta.json')
    meta_repo = FileSystem(url=metadata_fname)
    try:
        eval_config = meta_repo.load(EvaluationConfiguration())
    except IOError:
        print('Unable to load metadata file due to filesystem error or file not existing. Replotting everything by default.')
        eval_config = EvaluationConfiguration()

    if eval_config.n_cat is None or n_cat > eval_config.n_cat:
        force_plot_all = True

    # determine which data we are actually computing and whether the data should be shared
    active_data_products = {}
    for task_name, calc in data_products.items():
        version = eval_config.get_evaluation_version(task_name)
        if calc.version != version or force_plot_all:
            active_data_products[task_name] = calc

    # set 'calc' status on relevant items, we always share from pair[0] with pair[1]
    calc_pairs = [('l-test','arp-plot'),
                  ('m-test','mag-hist'),
                  ('l-test', 'carp-plot'),
                  ('prob-test','prob-plot')]

    # this should probably be a part of the class-state when we refactor the code
    print('Trying to determine if we can share calculation data between processing tasks...')
    for pair in calc_pairs:
        if set(pair).issubset(set(active_data_products.keys())):
            class_name0 = active_data_products[pair[0]].__class__.__name__
            class_name1 = active_data_products[pair[1]].__class__.__name__
            print(f'Found {class_name0} and {class_name1} in workload manifest that can share data, thus skipping calculations for {class_name1}.')
            active_data_products[pair[1]].calc = False

    # output some info for the user
    print(f'Will process {n_cat} catalogs from simulation\n')
    for k, v in active_data_products.items():
        print(f'Computing {v.__class__.__name__}')
    print('\n')

    if not name:
        days_since_mainshock = numpy.round(millis_to_days(origin_epoch - event_epoch))
        if u3etas_config['griddedOnly']:
            name = f'NoFaults, M{event.magnitude} + {days_since_mainshock} days'
        else:
            name = f'U3ETAS, M{event.magnitude} + {days_since_mainshock} days'

    # read the catalogs
    print('Begin processing catalogs', flush=True)
    t0 = time.time()
    loaded = 0
    u3 = load_stochastic_event_sets(filename=filename, type='ucerf3', name=name, region=aftershock_region)
    if not skip_processing:
        try:
            for i, cat in enumerate(u3):
                cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(aftershock_region).apply_mct(event.magnitude, event_epoch)
                for task_name, calc in active_data_products.items():
                    calc.process(copy.copy(cat_filt))
                tens_exp = numpy.floor(numpy.log10(i + 1))
                if (i + 1) % 10 ** tens_exp == 0:
                    t1 = time.time()
                    print(f'Processed {i+1} catalogs in {t1-t0} seconds', flush=True)
                if (i + 1) % n_cat == 0:
                    break
                loaded += 1
        except Exception as e:
            print(f'Failed loading at observed_catalog {i+1} with {str(e)}. This may happen normally if the simulation is incomplete\nProceeding to finalize plots')
            n_cat = loaded

        t2 = time.time()
        print(f'Finished processing catalogs in {t2-t0} seconds\n', flush=True)

        print('Processing catalogs again for distribution-based tests', flush=True)
        for k, v in active_data_products.items():
            if v.needs_two_passes == True:
                print(v.__class__.__name__)
        print('\n')

        # share data if needed
        print('Sharing data between related tasks...')
        for pair in calc_pairs:
            if set(pair).issubset(set(active_data_products.keys())):
                class_name0 = active_data_products[pair[0]].__class__.__name__
                class_name1 = active_data_products[pair[1]].__class__.__name__
                print(f'Sharing data from {class_name0} with {class_name1}.')
                active_data_products[pair[1]].data = active_data_products[pair[0]].data

        # old iterator is expired, need new one
        t2 = time.time()
        u3 = load_stochastic_event_sets(filename=filename, type='ucerf3', name=name, region=aftershock_region)
        for i, cat in enumerate(u3):
            cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(aftershock_region).apply_mct(event.magnitude, event_epoch)
            for task_name, calc in active_data_products.items():
                calc.process_again(copy.copy(cat_filt), args=(time_horizon, n_cat, end_epoch, comcat))
            # if we failed earlier, just stop there again
            tens_exp = numpy.floor(numpy.log10(i+1))
            if (i+1) % 10**tens_exp == 0:
                t3 = time.time()
                print(f'Processed {i + 1} catalogs in {t3 - t2} seconds', flush=True)
            if (i+1) % n_cat == 0:
                break

        # evaluate the catalogs and store results
        t1 = time.time()

        # make plot directory
        fig_dir = os.path.join(plot_dir, 'plots')
        mkdirs(fig_dir)

        # make results directory
        results_dir = os.path.join(plot_dir, 'results')
        if save_results:
            mkdirs(results_dir)

        # we want to
        for task_name, calc in active_data_products.items():
            print(f'Finalizing calculations for {task_name} and plotting')
            result = calc.post_process(comcat, args=(u3, time_horizon, end_epoch, n_cat))
            # plot, and store in plot_dir
            calc.plot(result, fig_dir, show=False)

            if save_results:
                # could expose this, but hard-coded for now
                print(f"Storing results from evaluations in {results_dir}", flush=True)
                calc.store_results(result, results_dir)

        t2 = time.time()
        print(f"Evaluated forecasts in {t2-t1} seconds", flush=True)

        # update evaluation config
        print("Updating evaluation metadata file", flush=True)
        eval_config.compute_time = utc_now_epoch()
        eval_config.catalog_file = catalog_fname
        eval_config.forecast_file = filename
        eval_config.forecast_name = name
        eval_config.n_cat = n_cat
        eval_config.eval_start_epoch = origin_epoch
        eval_config.eval_end_epoch = end_epoch
        eval_config.git_hash = current_git_hash()
        for task_name, calc in active_data_products.items():
            eval_config.update_version(task_name, calc.version, calc.fnames)
        # save new meta data
        meta_repo.save(eval_config.to_dict())

        # writing observed_catalog
        print(f"Saving ComCat observed_catalog used for Evaluation", flush=True)
        evaluation_repo = FileSystem(url=catalog_fname)
        evaluation_repo.save(comcat.to_dict())

        print(f"Finished evaluating everything in {t2-t0} seconds with average time per observed_catalog of {(t2-t0)/n_cat} seconds", flush=True)
    else:
        print('Skip processing flag enabled so skipping straight to report generation.')

    # create the notebook for results, but this should really be a part of the processing task as to support an arbitrary
    # set of inputs. right now this is hard-coded to support these types of analysis
    if generate_markdown:
        md = MarkdownReport('README.md')

        md.add_introduction(adict={'simulation_name': u3etas_config['simulationName'],
                                   'origin_time': epoch_time_to_utc_datetime(origin_epoch),
                                   'evaluation_time': epoch_time_to_utc_datetime(end_epoch),
                                   'catalog_source': 'ComCat',
                                   'forecast_name': 'UCERF3-ETAS',
                                   'num_simulations': n_cat})

        md.add_sub_heading('Visual Overview of Forecast', 1,
                "These plots show qualitative comparisons between the forecast "
                f"and the target data obtained from ComCat. Plots contain events within {numpy.round(millis_to_days(end_epoch-origin_epoch))} days "
                f"of the forecast start time and within {numpy.round(3*rupture_length/1000)} kilometers from the epicenter of the mainshock.  \n  \n"
                "All catalogs (synthetic and observed) are processed using the time-dependent magnitude of completeness model from Helmstetter et al., (2006).\n")


        md.add_result_figure('Cumulative Event Counts', 2, list(map(get_relative_path, eval_config.get_fnames('cum-plot'))), ncols=2,
                             text="Percentiles for cumulative event counts are aggregated within one-day bins. \n")

        md.add_result_figure('Magnitude Histogram', 2, list(map(get_relative_path, eval_config.get_fnames('mag-hist'))),
                             text="Forecasted magnitude number distribution compared with the observed magnitude number "
                                  "distribution from ComCat. The forecasted number distribution in each magnitude bin is "
                                  "shown using a box and whisker plot. The box indicates the 95th percentile range and the "
                                  "whiskers indicate the minimum and maximum values. The horizontal line indicates the median.\n")

        md.add_result_figure('Approximate Rate Density with Observations', 2, list(map(get_relative_path, eval_config.get_fnames('arp-plot'))), ncols=2,
                             text="The approximate rate density is computed from the expected number of events within a spatial cell and normalized over "
                                  "the time horizon of the forecast and the area of the spatial cell.\n")

        md.add_result_figure('Conditional Rate Density', 2, list(map(get_relative_path, eval_config.get_fnames('carp-plot'))), ncols=2,
                             text="Plots are conditioned on number of target events ± 5%, and can be used to create "
                                  "statistical tests conditioned on the number of observed events. In general, these plots will tend to "
                                  "be undersampled with respect to the entire distribution from the forecast.\n")

        md.add_result_figure('Spatial Probability Plot', 2,
                             list(map(get_relative_path, eval_config.get_fnames('prob-plot'))), ncols=2,
                             text="Probability of one or more events occuring in an individual spatial cell. This figure shows another way of "
                                  "visualizing the spatial distribution of a forecast.")

        md.add_sub_heading('CSEP Consistency Tests', 1, "<b>Note</b>: These tests are explained in detail by Savran et al., (In review).\n")

        md.add_result_figure('Number Test', 2, list(map(get_relative_path, eval_config.get_fnames('n-test'))),
                             text="The number test compares the earthquake counts within the forecast region aginst observations from the"
                                  " target observed_catalog.\n")

        md.add_result_figure('Magnitude Test', 2, list(map(get_relative_path, eval_config.get_fnames('m-test'))),
                             text="The magnitude test computes the sum of squared residuals between normalized "
                                  "incremental magnitude number distributions."
                                  " The test distribution is built from statistics scored between individal catalogs and the"
                                  " expected magnitude number distribution of the forecast.\n")

        md.add_result_figure('Likelihood Test', 2, list(map(get_relative_path, eval_config.get_fnames('l-test')['l-test'])),
                             text="The likelihood tests uses a statistic based on the continuous point-process "
                                  "likelihood function. We approximate the rate-density of the forecast "
                                  "by stacking synthetic catalogs in spatial bins. The rate-density represents the "
                                  "probability of observing an event selected at random from the forecast. "
                                  "Event log-likelihoods are aggregated for each event in the observed_catalog. This "
                                  "approximation to the continuous rate-density is unconditional in the sense that it does "
                                  "not consider the number of target events. Additionally, we do not include the magnitude component "
                                  "of the forecast to minimize the amount of undersampling present in these simulations.\n")

        md.add_result_figure('Probability Test', 2, list(map(get_relative_path, eval_config.get_fnames('prob-test'))),
                             text="This test uses a probability map to build the test distribution and the observed "
                                  "statistic. Unlike the pseudo-likelihood based tests, the test statistic is built "
                                  "by summing probabilities associated with cells where earthquakes occurred once. In effect,"
                                  "two simulations that have the exact same spatial distribution, but different numbers of events "
                                  "will product the same statistic.")

        md.add_result_figure('Spatial Test', 2, list(map(get_relative_path, eval_config.get_fnames('l-test')['s-test'])),
                             text="The spatial test is based on the same likelihood statistic from above. However, "
                                  "the scores are normalized so that differences in earthquake rates are inconsequential. "
                                  "As above, this statistic is unconditional.\n")

        md.add_sub_heading('One-point Statistics', 1, "")
        md.add_result_figure('B-Value Test', 2, list(map(get_relative_path, eval_config.get_fnames('bv-test'))),
                             text="This test compares the estimated b-value from the observed observed_catalog along with the "
                                  "b-value distribution from the forecast. This test can be considered an alternate form to the Magnitude Test.\n")

        md.add_sub_heading('Distribution-based Tests', 1, "")
        md.add_result_figure('Inter-event Time Distribution', 2, list(map(get_relative_path, eval_config.get_fnames('ietd-test'))),
                             text='This test compares inter-event time distributions based on a Kilmogorov-Smirnov type statistic '
                                  'computed from the empiricial CDF.\n')

        md.add_result_figure('Inter-event Distance Distribution', 2,
                             list(map(get_relative_path, eval_config.get_fnames('iedd-test'))),
                             text='This test compares inter-event distance distributions based on a Kilmogorov-Smirnov type statistic '
                                  'computed from the empiricial CDF.\n')

        md.add_result_figure('Total Earthquake Rate Distribution', 2, list(map(get_relative_path, eval_config.get_fnames('terd-test'))),
                             text='The total earthquake rate distribution provides another form of insight into the spatial '
                                  'consistency of the forecast with observations. The total earthquake rate distribution is computed from the '
                                  'cumulative probability distribution of earthquake occurrence against the earthquake rate per spatial bin.\n')

        md.finalize(plot_dir)

    t1 = time.time()
    print(f'Completed all processing in {t1-t0} seconds')


