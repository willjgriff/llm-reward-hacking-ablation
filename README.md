# Orthogonalization Against Reward Hacking

SPAR project page: https://sparai.org/projects/f26/rec6s8CRgbmKsDlZ1/
Project doc: https://docs.google.com/document/d/1jdNhWacPyFmWbkDv_EPhm6EUnsgtRVCdDqkH3u3729c/edit?usp=sharing
Project notes: https://docs.google.com/document/d/1I8rx2zz3LTnX0frJaJ5A-KGQvwkEE9rNF0wo-7VbLnc/edit?usp=sharing

## Overview

This project is an attempt at minimising reward hacking behaviour of models by orthogonalisation (removal via ablation) of reward hacking directions. We will compare our attempts to a baseline DPO model and evaluate how it compares in 2 ways: 

- It's tolerance to low-quality training data (how basic the set of reward hacking examples can be to remove advanced reward hacking behaviour)
- It's sample efficiency (how few examples do we need to remove reward hacking behaviour)

We will start with small models and if successful increase model size to match the available resources. 

