#! /usr/bin/env python

import matplotlib

import seaborn as sns
import argparse

from csep.core.analysis import ucerf3_consistency_testing


matplotlib.use('agg')
matplotlib.rcParams['figure.max_open_warning'] = 150
sns.set()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform CSEP consistency testing on UCERF3-ETAS forecasts.")
    required = parser.add_argument_group('required')
    optional = parser.add_argument_group('optional')
    required.add_argument("--sim_dir", help="Directory containing binary file of stochastic event sets.", required=True)
    required.add_argument("--eval_days", help="Number of days to evaluate forecast starting at the forecast start time", required=True)
    required.add_argument("--event_id", help="String corresponding to the ComCat event_id", required=True)
    optional.add_argument("--generate_markdown", action='store_true', help="True/False to generate markdown document. Defaults to true.")
    optional.add_argument("--plot_dir", default=None, help="Specifies directory to store plots can be different than the simulation directory.")
    optional.add_argument("--n_cat", default=None, type=int, help="Number of catalogs to process. Defaults to all catalogs.")
    optional.add_argument("--catalog_repo", default=None, help="Filepath to ComCat observed_catalog to use in Json format. Defaults to downloading.")
    optional.add_argument("--event_repo", default=None, help="Filepath for json representation of event definition. Defaults to downloading.")
    optional.add_argument("--save_results", action='store_true', help="True/False to store evaluation results as json documents.")
    optional.add_argument("--force_replot", action='store_true', help="True/False Force replotting of all figures.")
    optional.add_argument("--skip_processing", action='store_true', help="True/False flag to skip evaluations to generate report.")
    optional.add_argument("--name", default="", type=str, help="Name of the forecast does not have to be Unique")
    args = parser.parse_args()

    ucerf3_consistency_testing(args.sim_dir, args.event_id, args.eval_days,
                               n_cat=args.n_cat, plot_dir=args.plot_dir, generate_markdown=args.generate_markdown,
                               catalog_repo=args.catalog_repo, save_results=args.save_results, event_repo=args.event_repo,
                               skip_processing=args.skip_processing, force_plot_all=args.force_replot, name=args.name)
