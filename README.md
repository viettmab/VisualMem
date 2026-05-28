# VisualMem: Personal Visual Memory from Explicit and Implicit Evidence

###  [Project Page](https://viettmab.github.io/visualmem-page) | [arXiv](https://arxiv.org/abs/2605.28806) | [HF Datasets](https://huggingface.co/datasets/viettmab/visualmem)

---
**Personal Visual Memory from Explicit and Implicit Evidence**<br>
Viet Nguyen<sup>1</sup>, Thao Nguyen<sup>2</sup>, Vishal M. Patel<sup>1, ¶</sup>, Yuheng Li<sup>3, ¶</sup><br>
*<sup>1</sup>Johns Hopkins University, <sup>2</sup>University of Wisconsin-Madison, <sup>3</sup>Adobe Research*<br>

> Long-term memory is increasingly important for personalized AI agents, yet existing benchmarks and methods remain largely text-centric. Even when images are included, the user-specific information needed for later questions is typically recoverable from text alone, and most memory systems reduce image turns to generic captions. Yet images often carry personal information that text rarely states---both explicit evidence, such as recurring user-associated entities, and implicit evidence, such as latent user facts inferred from visual or multimodal cues. We introduce a benchmark for personal visual memory that targets both forms of evidence, and propose VisualMem, a hybrid visual--text architecture that augments a text-memory backend with a structured personal visual memory module. Rather than collapsing images into captions, VisualMem uses conversational context to resolve identity, ownership, and durable user facts. Experiments show that VisualMem substantially outperforms prior memory systems on our benchmark while remaining competitive on standard text-memory benchmarks, indicating that personal visual memory is a distinct and important component of long-term memory for personalized AI agents.

*(¶: equal advising)*

---

## Getting Started

Clone the repository and create a Python environment:

```bash
git clone https://github.com/viettmab/viusalmem.git
cd VisualMem

conda create -n visualmem python=3.12 -y
conda activate visualmem
pip install -U MemoryOS Pillow google-genai "qdrant-client[fastembed]"
```

Create a `.env` file from the example template, then fill in the required values:

```bash
cp .env-example .env
```

## Download Benchmark Data

The VisualMem benchmark data is hosted on Hugging Face: [VisualMem Benchmark Data](https://huggingface.co/datasets/viettmab/visualmem)

Download the benchmark archive, then unzip it into a local folder named
`visualmem_data`:

```bash
pip install "huggingface_hub[cli]"

hf download viettmab/visualmem visualmem_data.zip \
  --repo-type dataset \
  --local-dir .

unzip visualmem_data.zip
```

After downloading, the repository should look like this:

```text
VisualMem/
|-- visualmem_data/
|   |-- data.json
|   |-- images/
|   `-- viewer/
|-- evaluation/
|-- src/
`-- run.sh
```

The evaluation scripts use `visualmem_data/data.json` by default. If you store the benchmark elsewhere, pass `--input /path/to/data.json` to the individual evaluation modules.

## Evaluation

The full evaluation pipeline is wrapped by [`run.sh`](run.sh). It runs four stages:

1. Ingest persona conversations into memory.
2. Search memory for each benchmark question.
3. Generate multiple-choice answers.
4. Compute aggregate statistics.

Run the default VisualMem evaluation:

```bash
bash run.sh
```

Before launching a long run, you may want to edit the configuration block at the top of [`run.sh`](run.sh):

```bash
LIB="ours"
VERSION="window-full"
WORKERS=10
TOPK=20
```

Results are written under:

```text
results/visualmem/<LIB>-<VERSION>/
|-- memories/
|-- tmp/
|-- <LIB>_search_results.json
|-- <LIB>_responses.json
|-- <LIB>_statistics.json
`-- success_records.txt
```

## Citation

If you find VisualMem useful, please cite:

```bibtex
@article{nguyen2026visualmem,
  title={Personal Visual Memory from Explicit and Implicit Evidence},
  author={Nguyen, Viet and Nguyen, Thao and Patel, Vishal M and Li, Yuheng},
  journal={arXiv preprint arXiv:2605.28806},
  year={2026}
}
```
