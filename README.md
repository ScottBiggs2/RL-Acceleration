src/
├── models/
│   └── load_gemma.py        # load Gemma-3-270M + tokenizer
│
├── reward/
│   └── reward_functions.py  # check if model answer matches label
│
├── mask/
│   ├── activation_hooks.py  # forward hook utils
│   ├── cav.py               # compute concept vectors (high vs low reward)
│   ├── mask_stats.py        # rank neurons/heads, threshold, save masks
│   └── ablation_tests.py    # forward-pass ablations
│
├── data/
│   └── load_openr1.py      # dataset subset & tokenizer prep
│
├── utils/
│   └── logging_utils.py    # experiment configs & results logging
│
└── main_mask_finder.py      # orchestration script

