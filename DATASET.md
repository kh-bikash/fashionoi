# Dataset card

## Source

I use the official [Fashionpedia dataset](https://github.com/cvdfoundation/fashionpedia), introduced by Jia et al. at ECCV 2020. The project reports 48,825 fashion images, 27 main apparel categories, 19 apparel parts, and 294 fine-grained attributes.

The local experiment uses the official `val_test2020.zip` download. In this workspace it expands to:

```text
val_test2020/
  test/
    3,200 JPG images
```

The archive contains test images, not public per-image attribute or segmentation labels. The code therefore never treats filenames as labels and does not report fabricated test accuracy.

## Optional Fashionpedia annotations

The official repository separately provides:

- `instances_attributes_train2020.json`
- `instances_attributes_val2020.json`
- `info_test2020.json`

Train and validation instance JSON contains category IDs, fine-grained attribute IDs, segmentation, and bounding boxes. Pass an instance file to the indexer with `--annotations`; apparel boxes will be appended to the annotation-free heuristic crops. Test metadata does not contain the hidden garment labels.

## Assignment-specific coverage

Fashionpedia was designed for apparel segmentation and attributes, not balanced environment recognition. To address the assignment's separate environment axis, I run a transparent zero-shot curation audit across:

- environment: office, urban street, park, home;
- clothing: formal, casual, outerwear;
- color: black, white, red, blue, yellow, green, orange, purple, pink, brown, beige.

The audit allocates explicit quotas and saves the exact 1,000 selected relative paths. These assignments are semantic retrieval proxies and must not be confused with human scene labels. `reports/coverage_context_montage.jpg` makes the proxy quality visually inspectable.

## Reproducibility and leakage controls

- Raw images, archives, model weights, and vector indexes are excluded from Git.
- The selected filenames, coverage report, source links, and code are versioned.
- Final retrieval queries are not used to train model weights.
- Curation uses broad axis prompts rather than the five exact evaluation sentences.
- Quantitative MRR, Recall@10, and nDCG@10 require relevance judgments from the supplied template.

## License and attribution

Users should review the official Fashionpedia terms before redistribution. The Fashionpedia project states that its annotations and ontology are provided under CC BY 4.0; image rights and redistribution conditions should be checked through the official source.
