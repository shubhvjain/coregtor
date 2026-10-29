Running CoRegTor on multiple targets
====================================

The ``coregtor`` command line tool predicts co-regulators for many target genes at once. There are two ways to do this, and both share the same basic setup. Start with the :ref:`setup <multi-setup>`, then follow the method that fits your run.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Method
     - Commands
     - Best for
   * - 1. Direct run
     - ``coregtor run``
     - Small runs (about 50 targets or fewer, though the limit in not enforced by the tool). One command, no temp files, you get the final result file.
   * - 2. Bulk run
     - ``coregtor bulk``, ``status``, ``result``
     - Large runs and HPC clusters. Work is split into batches that can run in parallel and be resumed, and the intermediate files are kept so you can create more results later.



CLI Commands summary
---------------------

.. code-block:: text

   # Method 1
   coregtor run         --input FILE --result RESULT [--output DIR] [--rerun]

   # Method 2
   coregtor bulk        --input FILE [--batch N] [--batch-id NAME] [--rerun]
                        [--output DIR] [--temp DIR]
   coregtor status      --input FILE [--name RESULT] [--output DIR] [--temp DIR]
   coregtor result      --input FILE --name RESULT [--rerun] [--jobs N]
                        [--output DIR] [--temp DIR]
   coregtor consolidate --input FILE [--output DIR] [--temp DIR]


.. _multi-setup:

Setup 
------

This is same for both methods. 

Data preprocessing
~~~~~~~~~~~~~~~~~~

CoRegTor expects **prepared input**.  Do all of the input preprocessing beforehand, so that the files you give CoRegTor contain exactly the genes you want to use. You need:

* **An expression matrix** (CSV). Genes are columns, samples are rows.
* **A source file** (text). One source (regulator) gene per line. 
* **Your targets**, either as a text file with one gene per line, or as a list written directly in the input file.

Gene names must be in the same format in all three.

The input file
~~~~~~~~~~~~~~

The experiment is described by one input file, JSON. A minimal example, ``exp1.json``:

.. code-block:: json

   {
      "id": "exp1",
      "inputs": {
         "data": "path/to/gene_expression.csv",
         "source": "path/to/sources.txt",
         "target_file": "path/to/targets.txt"
      },
      "ensemble_model" :{
         "method":"rf",
         "options":{
            "max_depth": 5,
            "n_estimators":1000,
            "n_jobs":-1
         }
      },
      "context_generation":{
         "method": "tree_paths",
         "transform":"gene_frequency"
      },            
      "results":[
         {
            "id":"default",
            "distance":"chebyshev_distance",
            "method":"community_detection",
            "options":{
               "edge_creation_method": "threshold_percentile",
               "edge_creation_value": 0.75,
               "resolution": 0.75
            }
         }
      ]
   }
* This file specifies how to get the input data on which the tool will run and the pipeline options which are common for each target. The format is the same as the core  :func:`RunCoRegTor <coregtor.RunCoRegTor>`  method. 
* ``id`` names the experiment. Everything is written into a folder with this
  name.
* Input options (``input``) :
   * It is recommended to use absolute paths. Paths may contain environment variables and ``~``. For eg. ``"$HOME/data/expr.csv"``. 
   * Instead of ``target_file`` you can use ``"target_list": ["GENE1", "GENE2"]``.
* Ensemble model (``ensemble_model``) :
   * ``method`` describe which ensemble model to use.
   * ``options`` describes valid options for the selected ensemble model. 
   *  Check :func:`here <coregtor.create_model>`  for all possible values for both options. 
* Context generation (``context_generation``):
   * These settings describe how most predictive genes are selected and how the context for comparison is generated.
   * Check here for more details
* Result generation (``results``): 
   * Since the next step in the pipeline is quite flexible, a user can try multiple configurations of cluster generation.  
   * Each result needs an ``id``, ``measure`` (distance measure), ``method`` (clustering method to use) and optionally ``options`` (clustering options). See  :func:`RunCoRegTor <coregtor.RunCoRegTor>`  method for details description. 





Specifying the paths for results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You do not need to create any folders. They are created when first needed.

* **Results** go to ``<output>/<experiment id>/``. If you do not pass
  ``--output``, the base folder is ``~/coregtor/results``.
* **Temp files** (bulk method only) go to ``<temp>/<experiment id>/``. If you do
  not pass ``--temp``, the base folder is ``~/coregtor/temp``.

For ``exp1`` with the default folders:

.. code-block:: text

   ~/coregtor/
   ├── results/exp1/
   │   └── result_res1.csv        final result (both methods)
   └── temp/exp1/                 bulk method only
       ├── status.db              progress of every target
       ├── Batch-XXXXXXXX.db      scratch file of a running batch
       └── GENE1.pkl, ...         one temp file per target

The experiment ``id`` is always added as a subfolder, so several experiments can
share the same base folders. Both flags accept environment variables; quote them
so your shell does not expand them too early:

.. code-block:: bash

   --output '$HOME/runs/out'  --temp '$SCRATCH/tmp'

If a variable is not set, the tool stops with an error instead of creating a
folder with a literal ``$`` in its name.


Method 1: Direct run (small runs)
---------------------------------

Use this for roughly 50 targets or fewer, when you just want the final file.
Nothing except the result is written to disk.

.. code-block:: bash

   coregtor run --input exp1.json --result res1 --output '$HOME/coregtor/results'

``--result`` is the ``id`` of one entry in ``results``. ``--output`` is required
for this command.

For each target, in turn, CoRegTor trains the model, builds its contexts,
computes the distances and clusters the genes, then discards the model and moves
on to the next target. At the end all targets are combined into
``<output>/exp1/result_res1.csv``. Targets that fail are listed at the end and
left out of the file, while the rest still complete.

Options:

``--rerun``
   Overwrite the result file if it already exists. Without it, an existing file
   is left alone and the command stops.

What to expect:

* **No temp files**, so nothing to clean up and no disk space needed.
* **Nothing is saved along the way.** If the run is interrupted you start from
  scratch.
* **A different result needs a full recompute.** If you later want the same
  targets with another distance or clustering method, the models are trained
  again. Method 2 avoids this.
* Targets run one after another. Speed comes from the model itself, which uses
  the ``n_jobs`` setting in the ``ensemble_model`` options (all cores by
  default).

If your target list is large, or you expect to want several results from the
same models, use Method 2.


Method 2: Bulk run (large runs and HPC)
---------------------------------------

This method splits the work into two steps:

1. ``coregtor bulk`` trains a model for each target and saves a temp file per
   target. This is the slow step, and it can be spread over many jobs.
2. ``coregtor result`` turns the temp files into the final result file. This
   step is fast.

Because the temp files are kept, you can create more results later (different
distance or clustering settings) without retraining.

.. warning::

   Do not change the targets after the first ``bulk`` call. The list of targets
   is stored when you start, and a later change is rejected with an error. To
   start over, delete the experiment's temp folder.

Step 1: generate the temp files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   coregtor bulk --input exp1.json --batch 100 --temp '$SCRATCH/coregtor_temp'

What happens:

1. The data is loaded and checked.
2. On the very first call, ``status.db`` is created with every target marked
   *pending*.
3. Up to 100 pending targets are *claimed* for this batch.
4. A model is trained for each claimed target and its temp file is saved.
5. The outcome of the batch (*done* or *failed* per target) is written to
   ``status.db``.

Run the command again to process the next 100 targets. When nothing is left you
see ``No pending genes, nothing to do``.

Options:

``--batch N``
   Number of targets per call (default 100).

``--batch-id NAME``
   Name for the batch (default: a timestamp such as ``Batch-20260101-120000``).

``--rerun``
   Recompute temp files even if they already exist.

.. tip::

   Put ``--temp`` on fast local or scratch storage. The temp files can be large,
   and you can delete them once all the results you need are generated.

Running on an HPC cluster
~~~~~~~~~~~~~~~~~~~~~~~~~

Several ``bulk`` calls can run at the same time on the same experiment. Each one
claims its own targets, so none is processed twice. This fits a job array, for
example with SLURM:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --array=1-20
   #SBATCH --cpus-per-task=4

   coregtor bulk --input exp1.json --batch 50 --temp '$SCRATCH/coregtor_temp'

Things to keep in mind:

* **All jobs must use the same** ``--temp`` **and** ``--output`` **folders**,
  and these must be on storage that every node can reach.
* **Keep each batch short enough.** A claimed target that has not reported back
  after 2 hours is assumed to belong to a crashed job and is handed out again.
  If a batch takes longer than that, its targets may be processed twice. Choose
  ``--batch`` so one batch finishes well within 2 hours.
* **Mind the CPU settings.** By default the model uses all cores
  (``"n_jobs": -1``). If several jobs share a machine, set ``n_jobs`` in the
  ``ensemble_model`` options to the number of cores each job should use.
* The total number of jobs does not need to match the number of targets. If jobs
  finish early or some are never started, just submit more later; they pick up
  whatever is still pending.

Check progress
~~~~~~~~~~~~~~

.. code-block:: bash

   coregtor status --input exp1.json

Example output:

.. code-block:: text

   Experiment: exp1
     pending      400   40.0%
     claimed        0    0.0%
     done         590   59.0%
     failed        10    1.0%
     total       1000

* **pending**: not started yet.
* **claimed**: currently being processed by a batch.
* **done**: the temp file was created.
* **failed**: something went wrong. The first few reasons are printed below the
  table.

Add ``--name res1`` to also see whether that result file has been generated.
``status`` only reads ``status.db``, so it is fast and safe to run while batches
are still working.

Step 2: generate the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the targets are done, create the result:

.. code-block:: bash

   coregtor result --input exp1.json --name res1 --temp '$SCRATCH/coregtor_temp'

For each target this computes the distance matrix, clusters it with the method
from the input file, and combines everything into
``<output>/exp1/result_res1.csv``. Targets without a usable temp file (for
example failed ones) are skipped.

Options:

``--name NAME``
   The ``id`` of one entry in ``results`` (required).

``--rerun``
   Overwrite the result file if it already exists.

``--jobs N``
   Number of parallel workers for this step.

If some targets are not finished yet, you get a warning and the result covers
only the finished ones. It is fine to run ``result`` early to take a look, and
again later with ``--rerun``.

To create another result, add an entry to ``results`` in the input file (the
targets must stay the same) and run ``result`` with the new name. No model is
retrained.

Recovering from problems
~~~~~~~~~~~~~~~~~~~~~~~~

A batch crashed or was killed
   Its targets stay *claimed* and are re-queued automatically after 2 hours.
   Temp files that were already written are kept and reused. Targets the batch
   finished before dying may still show as *claimed* until you merge its
   progress file:

   .. code-block:: bash

      coregtor consolidate --input exp1.json

   ``status`` tells you when this is needed, with a message about batch
   databases that are not merged yet.

Some targets failed
   Failed targets are not retried automatically. First look at the reasons shown
   by ``status``. Once the cause is fixed, put them back in the queue and
   recompute them:

   .. code-block:: bash

      sqlite3 ~/coregtor/temp/exp1/status.db \
        "UPDATE genes SET status='pending', worker=NULL, error=NULL WHERE status='failed'"
      coregtor bulk --input exp1.json --rerun

   ``--rerun`` is required because a failed target leaves behind a temp file
   that records the failure, and without ``--rerun`` it would be skipped. Only
   the targets claimed in that call are recomputed.

"The targets in the input file no longer match status.db"
   You changed the targets after starting. Either restore the original targets,
   or delete the experiment's temp folder (``~/coregtor/temp/exp1``, or the
   equivalent under your ``--temp``) and start again.


