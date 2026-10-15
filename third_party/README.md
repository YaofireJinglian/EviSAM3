# Bundled upstream code

- `unilm/`: BEiT3 and TorchScale components from Microsoft UNILM. The identical
  nested `unilm/unilm/` copy has been removed. Existing source copyright notices
  and upstream documentation are retained.
- `sam3_eval/`: upstream SAM3 evaluation and dataset preparation utilities,
  previously located in `scripts/eval/`. These are separate from the EviSAM3
  evaluation tools in `tools/evaluation/`.

The integrated SAM3 source remains in `models/sam3/` because project model
builders import it directly. Its license and upstream documentation remain in
`models/`. These components retain their respective upstream license terms.
