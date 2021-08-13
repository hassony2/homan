#!/usr/bin/env python
# -*- coding: utf-8 -*-

SIDES = {
    "s1": "right",
    "s2": "left",
    "s3": "left",
    "s4": "right",
    "s5": "right",
    "s6": "right",
    "s7": "left",
    "s8": "right",
    "s9": "left",
    "s10": "right",
    "s11": "left",
}

MODELS = {
    # Mobile phones
    "o6": {
        "path": "7ea27ed05044031a6fe19ebe291582.obj",
        "scale": 0.07
    },
    "o8": {
        "path": "3ff176780a009cd93b61739f3c4d4342.obj",
        "scale": 0.08
    },
    "o9": {
        "path": "e55ef720305bfcac284432ce2f42f498.obj",
        "scale": 0.08
    },
    "o10": {
        "path": "d7ed512f7a7daf63772afc88105fa679.obj",
        "scale": 0.12
    },
    # Cans
    "o21": {
        "path": "3fd8dae962fa3cc726df885e47f82f16.obj",
        "scale": 0.2
    },
    "o22": {
        "path": "3fd8dae962fa3cc726df885e47f82f16.obj",
        "scale": 0.2
    },
    "o23": {
        "path": "3fd8dae962fa3cc726df885e47f82f16.obj",
        "scale": 0.2
    },
    "o24": {
        "path": "11c785813efc4b8630eaaf40a8a562c1.obj",
        "scale": 0.2
    },
    "o25": {
        "path": "11c785813efc4b8630eaaf40a8a562c1.obj",
        "scale": 0.2
    },
    # Remote controls
    "o46": {
        "path": "8e167ac56b1a437017d17fdfb5740281.obj",
        "scale": 0.2
    },
    "o47": {
        "path": "57759e351ec76d86d3c1501c166e6b2a.obj",
        "scale": 0.3
    },
    "o48": {
        "path": "a97a5e1c99e165c2327b86d5194a11a7.obj",
        "scale": 0.12
    },
    "o49": {
        "path": "a036b6be1c50f61fa046bbac53886364.obj",
        "scale": 0.3
    },
    "o50": {
        "path": "cc846e66cbfe697bffb5024c146ec04e.obj",
        "scale": 0.22
    },
    # Bulbs:
    "o16": {
        "path": "206ef4c97f50caa4a570c6c691c987a8.obj",
        "scale": 0.12,
    },
    "o17": {
        "path": "8338a18d589c26d21c648623457982d0.obj",
        "scale": 0.12,
    },
    "o18": {
        "path": "8338a18d589c26d21c648623457982d0.obj",
        "scale": 0.12,
    },
    "o19": {
        "path": "b0c346ea1fa3ad0b2d7dd0a148440b17.obj",
        "scale": 0.15,
    },
    # Ball
    "o31": {
        "form": "sphere",
        "scale": 0.025,
    },
    "o32": {
        "form": "sphere",
        "scale": 0.03,
    },
    "o34": {
        "form": "sphere",
        "scale": 0.06,
    },
    # Cups
    "o41": {
        "path": "d75af64aa166c24eacbe2257d0988c9c.obj",
        "scale": 0.13,
    },
    "o42": {
        "path": "61c10dccfa8e508e2d66cbf6a91063.obj",
        "scale": 0.12,
    },
    "o43": {
        "path": "3143a4accdc23349cac584186c95ce9b.obj",
        "scale": 0.13,
    },
    "o44": {
        "path": "9af98540f45411467246665d3d3724c.obj",
        "scale": 0.13,
    },
    "o45": {
        "path": "ea127b5b9ba0696967699ff4ba91a25.obj",
        "scale": 0.13,
    },
}

FULL_MODELS = {
    # Mobile phones
    "o6": {
        "path":
        "02992529/7ea27ed05044031a6fe19ebe291582/models/model_normalized_proc.obj",
        "scale": 0.07
    },
    "o8": {
        "path":
        "02992529/3ff176780a009cd93b61739f3c4d4342/models/model_normalized_proc.obj",
        "scale": 0.08
    },
    "o9": {
        "path":
        "02992529/e55ef720305bfcac284432ce2f42f498/models/model_normalized_proc.obj",
        "scale": 0.08
    },
    "o10": {
        "path":
        "02992529/d7ed512f7a7daf63772afc88105fa679/models/model_normalized_proc.obj",
        "scale": 0.12
    },
    # Cans
    "o21": {
        "path":
        "02946921/3fd8dae962fa3cc726df885e47f82f16/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    "o22": {
        "path":
        "02946921/3fd8dae962fa3cc726df885e47f82f16/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    "o23": {
        "path":
        "02946921/3fd8dae962fa3cc726df885e47f82f16/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    "o24": {
        "path":
        "02946921/11c785813efc4b8630eaaf40a8a562c1/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    "o25": {
        "path":
        "02946921/11c785813efc4b8630eaaf40a8a562c1/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    # Remote controls
    "o46": {
        "path":
        "04074963/8e167ac56b1a437017d17fdfb5740281/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    "o47": {
        "path":
        "04074963/57759e351ec76d86d3c1501c166e6b2a/models/model_normalized_proc.obj",
        "scale": 0.3
    },
    "o48": {
        "path":
        "04074963/a97a5e1c99e165c2327b86d5194a11a7/models/model_normalized_proc.obj",
        "scale": 0.12
    },
    "o49": {
        "path":
        "04074963/a036b6be1c50f61fa046bbac53886364/models/model_normalized_proc.obj",
        "scale": 0.3
    },
    "o50": {
        "path":
        "04074963/cc846e66cbfe697bffb5024c146ec04e/models/model_normalized_proc.obj",
        "scale": 0.22
    },
    # Bulbs:
    "o16": {
        "path":
        "03636649/206ef4c97f50caa4a570c6c691c987a8/models/model_normalized_proc.obj",
        "scale": 0.12,
    },
    "o17": {
        "path":
        "03636649/8338a18d589c26d21c648623457982d0/models/model_normalized_proc.obj",
        "scale": 0.12,
    },
    "o18": {
        "path":
        "03636649/8338a18d589c26d21c648623457982d0/models/model_normalized_proc.obj",
        "scale": 0.12,
    },
    "o19": {
        "path":
        "03636649/b0c346ea1fa3ad0b2d7dd0a148440b17/models/model_normalized_proc.obj",
        "scale": 0.15,
    },
    # Ball
    "o31": {
        "form": "sphere",
        "scale": 0.025,
    },
    "o32": {
        "form": "sphere",
        "scale": 0.03,
    },
    "o34": {
        "form": "sphere",
        "scale": 0.06,
    },
    # Cups
    "o41": {
        "path":
        "03797390/d75af64aa166c24eacbe2257d0988c9c/models/model_normalized_proc.obj",
        "scale": 0.13,
    },
    "o42": {
        "path":
        "03797390/61c10dccfa8e508e2d66cbf6a91063/models/model_normalized_proc.obj",
        "scale": 0.12,
    },
    "o43": {
        "path":
        "03797390/3143a4accdc23349cac584186c95ce9b/models/model_normalized_proc.obj",
        "scale": 0.13,
    },
    "o44": {
        "path":
        "03797390/9af98540f45411467246665d3d3724c/models/model_normalized_proc.obj",
        "scale": 0.13,
    },
    "o45": {
        "path":
        "03797390/ea127b5b9ba0696967699ff4ba91a25/models/model_normalized_proc.obj",
        "scale": 0.13,
    },
}
