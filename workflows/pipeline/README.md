# pipeline

Shell-script orchestration for driving the full `egt` analysis pipeline
end-to-end on a cluster. Set paths via `config.template.yaml` (copy to
`config.yaml`) and export them as environment variables.

Expected inputs:

- `RBH_DIR` — per-species RBH files from `odp`
- `SAMPLEDF_TSV` — sample-metadata dataframe
- `TREE_INFO` — `node_information.tsv` emitted by `egt newick-to-common-ancestors`
- `ALG_RBH` — the BCnSSimakov2022 ALG RBH file

Each `run_*.sh` wraps a single stage. Copy `config.template.yaml` to
`config.yaml`, edit paths for your environment, and run stages in order.
