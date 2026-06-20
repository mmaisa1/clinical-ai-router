# app/labels.py
"""
Label mapping for the 32 TCGA cancer types.

Maps short codes (used internally, in the database, and in API responses)
to verbose names (used in NLI hypothesis templates and LLM prompts).

This must stay in sync with research/01_train_and_evaluate.ipynb —
the model was trained and evaluated against these exact 32 labels.

Note: TCGA metadata defines 33 cancer types. LAML (acute myeloid leukemia)
has no corresponding pathology report text in this dataset and was excluded
during training. This system cannot classify LAML cases — it was never
trained on them. Extending support would require sourcing AML pathology
report text, retraining Stage 1, and re-running the full offline evaluation.
"""

LABEL_MAP = {
    "ACC":  "adrenocortical carcinoma",
    "BLCA": "bladder urothelial carcinoma",
    "BRCA": "breast invasive carcinoma",
    "CESC": "cervical squamous cell carcinoma and endocervical adenocarcinoma",
    "CHOL": "cholangiocarcinoma",
    "COAD": "colon adenocarcinoma",
    "DLBC": "diffuse large B-cell lymphoma",
    "ESCA": "esophageal carcinoma",
    "GBM":  "glioblastoma multiforme",
    "HNSC": "head and neck squamous cell carcinoma",
    "KICH": "kidney chromophobe",
    "KIRC": "kidney renal clear cell carcinoma",
    "KIRP": "kidney renal papillary cell carcinoma",
    "LGG":  "brain lower grade glioma",
    "LIHC": "liver hepatocellular carcinoma",
    "LUAD": "lung adenocarcinoma",
    "LUSC": "lung squamous cell carcinoma",
    "MESO": "mesothelioma",
    "OV":   "ovarian serous cystadenocarcinoma",
    "PAAD": "pancreatic adenocarcinoma",
    "PCPG": "pheochromocytoma and paraganglioma",
    "PRAD": "prostate adenocarcinoma",
    "READ": "rectum adenocarcinoma",
    "SARC": "sarcoma",
    "SKCM": "skin cutaneous melanoma",
    "STAD": "stomach adenocarcinoma",
    "TGCT": "testicular germ cell tumors",
    "THCA": "thyroid carcinoma",
    "THYM": "thymoma",
    "UCEC": "uterine corpus endometrial carcinoma",
    "UCS":  "uterine carcinosarcoma",
    "UVM":  "uveal melanoma",
}

# Reverse map — verbose name to code, used when parsing NLI output
VERBOSE_TO_CODE = {v: k for k, v in LABEL_MAP.items()}

# All valid codes, used for validating LLM fallback output
VALID_CODES = set(LABEL_MAP.keys())