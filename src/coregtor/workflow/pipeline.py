import joblib
import jsonschema
import pandas as pd
import traceback
from pathlib import Path
import os
from joblib import Parallel, delayed
import copy

from coregtor.context import create_context, transform_context, compare_context

from coregtor.clusters import identify_coregulators, get_cluster_method_list
from coregtor.context import get_distance_measures_list
from coregtor import generate_context

SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "inputs": {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "source": {"type": "string"},
                "target_list": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "target_file": {"type": "string"}
            },
            "required": ["data", "source"]
        },
        "ensemble_model": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["rf", "et"], "default": "rf"},
                "options": {"type": "object", "default": {"max_depth": 5, "n_estimators": 1000, "n_jobs": -1}}
            },
            "default": {"method": "rf", "options": {"max_depth": 5, "n_estimators": 1000, "n_jobs": -1}}
        },
        "context_generation": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "default": "tree_paths"},
                "transform": {"type": "string", "default": "gene_frequency"}
            },
            "default": {"method": "tree_paths", "transform": "gene_frequency"}
        },
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "distance", "method"],
                "properties": {
                    "id": {"type": "string"},
                    "distance": {
                        "type": "string",
                        "enum": get_distance_measures_list(),
                        "default": "cosine_distance"
                    },
                    "method": {
                        "type": "string",
                        "enum": get_cluster_method_list(),
                        "default": "hierarchical"
                    },
                    "options": {"type": "object", "default": {}}
                },
                "additionalProperties": True
            },
            "default":[{"id":"default","distance":"chebyshev_distance","method":"community_detection"}]
        }
    },
    "required": ["id", "inputs"]
}


def apply_defaults(data, schema):
    if not isinstance(schema, dict):
        return data

    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            return data
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key not in data:
                if "default" in prop_schema:
                    data[key] = prop_schema["default"]
            else:
                data[key] = apply_defaults(data[key], prop_schema)

    elif schema_type == "array":
        if not isinstance(data, list):
            return data
        item_schema = schema.get("items", {})
        data = [apply_defaults(item, item_schema) for item in data]

    return data


def get_validated_options(options):
    jsonschema.validate(instance=options, schema=SCHEMA)

    result_ids = [r["id"] for r in options.get("results", []) if "id" in r]
    if len(result_ids) != len(set(result_ids)):
        duplicates = {rid for rid in result_ids if result_ids.count(rid) > 1}
        raise jsonschema.ValidationError(
            f"Duplicate result ids found: {duplicates}")

    result = copy.deepcopy(options)
    apply_defaults(result, SCHEMA)
    return result

def read_gene_list(path):
    """Read one gene name per line, ignoring blank lines."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_inputs(config):
    """Return (expression_data, source_genes, targets) from a validated config."""
    inputs = config["inputs"]
    expression_data = pd.read_csv(inputs["data"], index_col=0)
    source_genes = read_gene_list(inputs["source"])

    if inputs.get("target_list"):
        targets = inputs["target_list"]
    elif inputs.get("target_file"):
        targets = read_gene_list(inputs["target_file"])
    else:
        raise ValueError("Provide either target_list or target_file in inputs")

    targets = list(dict.fromkeys(targets))  # drop duplicates, keep order
    return expression_data, source_genes, targets


def make_folder(base, exp_id):
    """Create and return <base>/<exp_id>. `base` may contain ~ or $VARS."""
    folder = Path(os.path.expandvars(base)).expanduser() / exp_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def find_result_config(config, result_id):
    """Return the entry of config["results"] with this id."""
    for r in config["results"]:
        if r["id"] == result_id:
            return r
    raise ValueError(f"Result configuration with id '{result_id}' not found.")


def sources_for(source_genes, target):
    """Source genes usable for this target (a gene is not its own regulator)."""
    return [g for g in source_genes if g != target]



class CoRegTorBulkPipeline:
    """Run the CoRegTor workflow for a batch of target genes.

    The pipeline follows a simple end-to-end workflow:

    1. Validate the user configuration against the project schema.
    2. Load the expression matrix and the source-gene list from the input files.
    3. For each target gene, build a model using the configured ensemble method.
    4. Extract decision paths from the trained model.
    5. Generate context sets from those paths and transform them into a gene-frequency matrix.
    6. Compare gene contexts using the configured distance metric.
    7. Cluster genes into co-regulator groups with the selected clustering method.
    8. Save intermediate checkpoints to a temp directory and write final cluster CSV files to an output directory.

    Parameters
    ----------
    config_dict : dict
        Configuration for the full experiment. This must match the schema defined in ``SCHEMA``. The required top-level keys are:

        - ``id``: experiment identifier
        - ``inputs``: input paths with ``data`` and ``source`` required; either ``target_list`` or ``target_file`` must also be supplied
        - ``ensemble_model``: optional ensemble settings
        - ``context_generation``: optional context-building settings; 
        - ``results``: list of result definitions, where each item must include ``id``, ``distance``, and ``method``

    run_config : dict, optional
        Runtime settings for file handling and execution behavior. The class reads
        these keys if present:

        - ``temp_path``: base folder for temporary checkpoints; default is ``./temp``
        - ``output_path``: base folder for generated CSV outputs; default is ``./output``
        - ``rerun``: if True, regenerate temporary checkpoints even when they exist and overwrite existing result CSV files
        - ``n_jobs``: number of parallel jobs used when generating result CSVs
 
    """

    def __init__(self, config_dict, run_config=None):
        self.config = get_validated_options(config_dict)
        self.run_config = run_config or {}
        self.exp_id = self.config["id"]

        self.temp_dir = make_folder(self.run_config.get("temp_path", "./temp"), self.exp_id)
        self.output_dir = make_folder(self.run_config.get("output_path", "./output"), self.exp_id)

        self.expression_data, self.source_genes, self.targets = load_inputs(self.config)

        if not any(g in self.expression_data.columns for g in self.source_genes):
            raise ValueError("No valid feature columns found in expression data")

    def _checkpoint_file(self, target):
        return self.temp_dir / f"{target}.pkl"

    def process_single_target_temp(self, target):
        """Train the model for one target and save its checkpoint.

        Returns True if a valid checkpoint exists afterwards, False if the
        target failed. A failed target leaves no checkpoint, so it is retried
        on the next run.
        """
        ckpt = self._checkpoint_file(target)
        if ckpt.exists() and not self.run_config.get("rerun", False):
            return True

        model_cfg = self.config["ensemble_model"]
        try:
            model, fi, paths, transformed, time_taken = generate_context(
                ge_data=self.expression_data,
                source_gene=sources_for(self.source_genes, target),
                target_gene=target,
                ensemble_method=model_cfg["method"],
                ensemble_options=copy.deepcopy(model_cfg["options"]),
            )
        except Exception:
            print(traceback.format_exc())
            ckpt.unlink(missing_ok=True)  # remove an old checkpoint when rerunning
            return False

        checkpoint = {
            "success": True,
            "model": model,
            "feature_importance": fi,
            "paths": paths,
            "transformed_context": transformed,
            "time_taken": time_taken,
        }
        joblib.dump(checkpoint, ckpt, compress=3)
        return True

    def process_single_target_result(self, target, res_config):
        ckpt_file = self._checkpoint_file(target)
        if not ckpt_file.exists():
            return None

        try:
            data = joblib.load(ckpt_file)
            if not data.get("success", False):
                return None

            ctx_cfg = self.config["context_generation"]
            sim_matrix = compare_context(
                data["transformed_context"],
                method=res_config["distance"],
                transformation_type=ctx_cfg["transform"],
            )
            clusters_df = identify_coregulators(
                distance_matrix=sim_matrix,
                target_gene=target,
                method=res_config["method"],
                options=res_config.get("options", {}),
                feature_importance=data["feature_importance"],
            )
            if clusters_df is None or clusters_df.empty:
                return None

            # clusters_df["note"] = f"{self.exp_id}-coregtor-{res_config['id']}"
            return clusters_df
        except Exception:
            return None

    def generate_temp_files(self):
        """Helper to run temp generation for all targets sequentially."""
        for target in self.targets:
            self.process_single_target_temp(target)

    def generate_results(self, result_id=None):
        """Build the result CSV(s) from the checkpoints, one target per parallel job."""
        n_jobs = self.run_config.get("n_jobs", -1)
        rerun = self.run_config.get("rerun", False)

        if result_id is None:
            configs_to_run = self.config["results"]
        else:
            configs_to_run = [find_result_config(self.config, result_id)]

        available = {f.stem for f in self.temp_dir.glob("*.pkl")}
        targets = [t for t in self.targets if t in available]

        for res_config in configs_to_run:
            out_file = self.output_dir / f"result_{res_config['id']}.csv"
            if out_file.exists() and not rerun:
                continue

            all_results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_single_target_result)(target, res_config)
                for target in targets
            )
            tables = [df for df in all_results if df is not None and not df.empty]

            if tables:
                pd.concat(tables, ignore_index=True).to_csv(out_file, index=False)



class CoRegTorRunPipeline:
    """Direct run: no temp files, only the final result CSV is written.

    Each target goes through RunCoRegTor and only its small cluster table is kept.
    """

    def __init__(self, config_dict, run_config=None):
        self.config = get_validated_options(config_dict)
        self.run_config = run_config or {}
        self.exp_id = self.config["id"]

        self.output_dir = make_folder(self.run_config.get("output_path", "./output"), self.exp_id)
        self.expression_data, self.source_genes, self.targets = load_inputs(self.config)

    def get_result_config(self, result_id):
        return find_result_config(self.config, result_id)

    def result_file(self, result_id):
        return self.output_dir / f"result_{result_id}.csv"

    def run_target(self, target, res_config):
        """Co-regulators of one target as a DataFrame, or None if there are none.
        Errors are raised, the caller decides what to do with them."""
        from coregtor import RunCoRegTor  # local import avoids a circular import

        model_cfg = self.config["ensemble_model"]
        results, _ = RunCoRegTor(  # second value (intermediates) is not needed
            ge_data=self.expression_data,
            source_gene=sources_for(self.source_genes, target),
            target_gene=target,
            ensemble_method=model_cfg["method"],
            ensemble_options=copy.deepcopy(model_cfg["options"]),
            distance_measure=res_config["distance"],
            cluster_method=res_config["method"],
            cluster_options=copy.deepcopy(res_config.get("options")),
        )
        if results is None or results.empty:
            return None
        return results

    def save_result(self, result_id, tables):
        out_file = self.result_file(result_id)
        pd.concat(tables, ignore_index=True).to_csv(out_file, index=False)
        return out_file