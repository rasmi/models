# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Beam data processing pipeline for AstroNet.

Beam data processing pipeline for AstroNet which reads raw data from GCS,
applies transformations, and writes outputs to TFRecords on GCS.

Based on generate_input_records.py:
https://github.com/tensorflow/models/blob/master/research/astronet/astronet/data/generate_input_records.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
import numpy as np
import pandas as pd

_DR24_TCE_URL = ('https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/'
                 'nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce')

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'project_id',
    None,
    help='GCP project ID to read from and write to.')
flags.DEFINE_string(
    'input_tce_csv_file',
    'https://storage.googleapis.com/kepler-data/q1_q17_dr24_tce.csv',
    help='CSV file containing the Q1-Q17 DR24 Kepler TCE table. '
    'Must contain columns: rowid, kepid, tce_plnt_num, tce_period, '
    'tce_duration, tce_time0bk, av_training_set. '
    'Will use Google-mirrored file by default. '
    'Alternatively, download from: %s ' % _DR24_TCE_URL)
flags.DEFINE_string(
    'input_tce_table',
    None,
    help='BigQuery table name containing the Q1-Q17 DR24 Kepler TCE table. '
    'Must contain columns: rowid, kepid, tce_plnt_num, tce_period, '
    'tce_duration, tce_time0bk, av_training_set. '
    'If set, this will take precedence over --input_tce_csv_file.')
flags.DEFINE_string(
    'kepler_data_dir',
    'gs://kepler-data/lightcurves',
    help='Base folder containing Kepler data. '
    'Will use Google-mirrored data by default.')
flags.DEFINE_string(
    'output_dir',
    None,
    help='Directory in which to save the output.')
flags.DEFINE_string(
    'runner',
    'direct',
    help='Beam runner (`direct` or `dataflow`).')
flags.mark_flag_as_required('output_dir')


def validate_project_id_flag(inputs):
  needs_project_id = bool('dataflow' in inputs['runner'].lower()
                          or inputs['input_tce_table'])
  return bool(inputs['project_id']) if needs_project_id else True

flags.register_multi_flags_validator(
    ['project_id', 'input_tce_table', 'runner'],
    validate_project_id_flag,
    message='--project_id must be set if running on Dataflow or '
    'reading from BigQuery.')


def read_tces(input_tce_csv_file, input_tce_table=None, project_id=None):
  """Read, filter, and partition a table of Kepler KOIs.

  Args:
    input_tce_csv_file: CSV file containing the Q1-Q17 DR24 Kepler TCE table.
    input_tce_table: BQ table name containing the Q1-Q17 DR24 Kepler TCE table.
    project_id: GCP project ID. Required if `input_tce_table` is passed.
  Returns:
    A `dict` with keys ['train', 'val', 'test'], where the values are lists of
      single `dict` TCE records as belonging to that subset.
  """
  # Name and values of the column in the input table to use as training labels.
  LABEL_COLUMN = 'av_training_set'
  ALLOWED_LABELS = {'PC', 'AFP', 'NTP'}
  # Total TCE count to verify table was read correctly.
  TOTAL_TCE_COUNT = 20367

  def read_tces_from_csv(filename):
    """Read a table of Kepler KOIs from a CSV file.

    Args:
      filename: TCE CSV filename.
    Returns:
      pd.DataFrame of TCEs.
    """
    tce_table = pd.read_csv(filename, comment='#')

    return tce_table

  def read_tces_from_bq(input_tce_table, project_id):
    """Read a table of Kepler KOIs from BigQuery.

    Args:
      input_tce_table: BigQuery table name containing TCE data.
      project_id: GCP project ID containing input TCE table.
    Returns:
      pd.DataFrame of TCEs.
    """
    query = 'SELECT * from `{}`'.format(input_tce_table)
    tce_table = pd.read_gbq(query, project_id=project_id, dialect='standard')

    return tce_table

  # Read table of Kepler KOIs from BigQuery table or CSV file.
  if input_tce_table:
    tce_table = read_tces_from_bq(input_tce_table, project_id)
  else:
    tce_table = read_tces_from_csv(input_tce_csv_file)

  tce_table = tce_table.set_index('rowid').sort_index()

  tce_table['tce_duration'] /= 24.0  # Convert hours to days.

  assert len(tce_table) == TOTAL_TCE_COUNT, (
      'Incorrect number of TCEs in table.')
  logging.info('Read TCE table with %d rows.', len(tce_table))

  # Filter TCE table to only include trainable entries.
  allowed_tces = tce_table[LABEL_COLUMN].isin(ALLOWED_LABELS)
  tce_table = tce_table[allowed_tces]
  num_tces = len(tce_table)
  logging.info('Filtered to %d TCEs with labels in %s.', num_tces,
               list(ALLOWED_LABELS))

  # Randomly shuffle the TCE table.
  np.random.seed(123)
  tce_table = tce_table.iloc[np.random.permutation(num_tces)]
  logging.info('Randomly shuffled TCEs.')

  # Partition the TCE table as follows:
  #   train_tces = 80% of TCEs
  #   val_tces = 10% of TCEs (for validation during training)
  #   test_tces = 10% of TCEs (for final evaluation)
  train_cutoff = int(0.80 * num_tces)
  val_cutoff = int(0.90 * num_tces)
  train_tces = tce_table[0:train_cutoff]
  val_tces = tce_table[train_cutoff:val_cutoff]
  test_tces = tce_table[val_cutoff:]
  logging.info(
      'Partitioned %d TCEs into training (%d), validation (%d) and test (%d)',
      num_tces, len(train_tces), len(val_tces), len(test_tces))

  tces = {'train': train_tces, 'val': val_tces, 'test': test_tces}
  tces = {subset: data.to_dict(orient='records')
          for subset, data in tces.items()}

  return tces


class ProcessTCEs(beam.DoFn):
  """Read and process a Kepler TCE light curve into a tf.Example."""

  def start_bundle(self):
    from astronet.data import preprocess  # pylint: disable=g-import-not-at-top
    self.preprocess = preprocess

  def process(self, tce, kepler_data_dir):
    """Read and process a Kepler TCE light curve into a tf.Example.

    Args:
      tce: `dict` record of a single TCE object, containing
        'kepid', 'tce_period', 'tce_duration', and 'tce_time0bk'.
      kepler_data_dir: Base folder containing Kepler data.
    Yields:
      String-serialized tf.Example proto of processed TCE light curve.
    """

    all_time, all_flux = self.preprocess.read_light_curve(tce['kepid'],
                                                          kepler_data_dir)
    time, flux = self.preprocess.process_light_curve(all_time, all_flux)
    example = self.preprocess.generate_example_for_tce(time, flux, tce)
    serialized_example = example.SerializeToString()

    yield serialized_example


def create_beam_pipeline():
  """Creates and returns a Beam pipeline based on FLAGS.

  Returns:
    A beam.Pipeline object.
  """
  # Define Beam pipeline options.
  options = {
      'runner': FLAGS.runner
  }
  # Define Dataflow-specific options.
  if 'dataflow' in FLAGS.runner.lower():
    temp_location = os.path.join(FLAGS.output_dir.rstrip('/'), 'tmp')
    options.update({
        'project': FLAGS.project_id,
        'job_name': 'astronet-preprocess-{}'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        'temp_location': temp_location,
        'max_num_workers': 5,
        'region': 'us-east1',
        'setup_file':
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../../', 'setup.py'))
    })
  pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)

  pipeline = beam.Pipeline(options=pipeline_options)

  return pipeline


def main(argv):
  del argv  # Unused.
  logging.set_verbosity(logging.INFO)

  pipeline = create_beam_pipeline()

  tces = read_tces(FLAGS.input_tce_csv_file,
                   FLAGS.input_tce_table, project_id=FLAGS.project_id)

  for subset, data in tces.items():
    subset_name = subset.title()
    _ = (pipeline
         | 'ReadTCEs{}'.format(subset_name) >> beam.Create(data)
         | 'ProcessTCEs{}'.format(subset_name) >> beam.ParDo(
             ProcessTCEs(), FLAGS.kepler_data_dir)
         | 'WriteTCEs{}'.format(subset_name) >> beam.io.WriteToTFRecord(
             os.path.join(FLAGS.output_dir.rstrip('/'), subset),
             file_name_suffix='.tfrecord'))

  pipeline.run().wait_until_finish()

  logging.info('Preprocessing complete.')

if __name__ == '__main__':
  app.run(main)
