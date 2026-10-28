# Third-party models

govkadr's source code is MIT-licensed. The model weights it downloads (`govkadr fetch-models`) are
**not** part of this repository and come with their own licences, which govern what you may do with
them and with outputs that depend on them. All are pinned to an exact upstream revision or SHA-256
in [`govkadr/weights.py`](govkadr/weights.py).

| Model | Used by | Source | Licence |
|---|---|---|---|
| MediaPipe Face Landmarker (478-point mesh, blendshapes) | all modes | Google | Apache-2.0 |
| BiRefNet_dynamic-matting | all modes | ZhengPeng7/BiRefNet_dynamic-matting | MIT |
| InsightFace `buffalo_l` (SCRFD-10GF + ArcFace R50) | identity audit | deepinsight/insightface | **non-commercial research only** |
| IC-Light `iclight_sd15_fc` | studio-ai | lllyasviel/ic-light | Apache-2.0 |
| Realistic Vision 5.1 (IC-Light base) | studio-ai | stablediffusionapi/realistic-vision-v51 | CreativeML OpenRAIL-M |
| FLUX.2 [klein] 4B | studio-flux, edge fill | black-forest-labs/FLUX.2-klein-4B | Apache-2.0 |

Notes:

* The identity audit uses InsightFace models whose licence restricts them to non-commercial research.
  Commercial use of the audit needs a different face-recognition model.
* The CreativeML OpenRAIL-M licence carries use-based restrictions; read it before redistributing
  IC-Light outputs.
* Only `strict-gov` output is intended for official ID or passport applications. The generative
  modes alter the image and are for studio / research use.
