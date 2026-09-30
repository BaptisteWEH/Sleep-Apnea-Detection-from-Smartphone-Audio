# Sleep Apnea Detection from Smartphone Audio

Can a smartphone on the nightstand detect sleep apnea? Published models report 88 to 89% accuracy, but they are trained on hospital recordings with professional microphones. This project tests what happens on real smartphone audio, and explains why every model I tried hits the same ceiling.

**Key finding:** the pre-trained audio representations identify *which patient* is speaking with **97.4% accuracy** (chance = 2%). The model learns the person before it learns the pathology.

[Full report (PDF, in French)](rapport_apnee_sommeil_1.pdf) · [Kaggle notebook](https://www.kaggle.com/code/baptiste05/notebook67e7528c95)

---

## Data

- **Dataset:** Tao et al. (2025), *Nature Scientific Data*. The only public dataset recorded by a real smartphone on the nightstand, with labels validated by simultaneous polysomnography.
- **Size:** 48 patients, about 7 hours of audio each (~340 h in total).
- The audio is not redistributed in this repository. Download it from the original source: LIEN_DATASET

## Method

- **Segments:** 40 s windows, shifted +5 s after the end of each event, to capture breathing before the apnea, the apnea itself and the recovery gasp (window length taken from Han et al., 2026).
- **Features:** mel-spectrograms, or YAMNet embeddings for the transfer learning models.
- **Patient-level split:** 32 patients for training, 8 for validation, 8 for testing. No patient ever appears in two sets. Without this, scores go above 90% and collapse on new patients.
- **Balancing per patient:** 50% apnea, 50% normal segments for each patient. Without it, a model reaches 85% by always predicting the majority class.
- **Training:** TensorFlow / Keras on Kaggle GPUs (T4 x2).

## Results

All results are on the 8 test patients, never seen during training.

| Model | Idea | Accuracy | AUC |
|---|---|---|---|
| CNN on mel-spectrogram | Visual patterns in the spectrogram | 65% | 0.67 |
| YAMNet transfer learning | Google model pre-trained on 2M sounds | 67% | 0.73 |
| YAMNet + adversarial identity head | Penalized when it recognizes the patient | 67% | 0.72 |
| CNN + BiLSTM | Temporal reading of the spectrogram | 66% | 0.73 |

<!-- ![Results](figures/results.png) -->

Every architecture plateaus at 65 to 67%. Changing the model does not help, so the bottleneck is upstream.

## Diagnosis: the model learns the patient, not the apnea

I kept the YAMNet embeddings and changed the target: instead of "apnea or not", predict **which of the 48 patients** produced the sound.

- **Result: 97.4% accuracy** (chance = 2%).
- The representation encodes voice, room acoustics and phone position much more strongly than the medical signal.
- A domain-adversarial head reduced patient identification from **53% to 12%**, but apnea detection did not improve accordingly.

<!-- ![Patient identity probe](figures/identity_probe.png) -->

**Takeaway:** generic pre-trained audio encoders (YAMNet, HuBERT, wav2vec) were built to tell people apart. In clinical audio, that goal works against the task.

## Open questions

- **Patient-invariant representations:** how much individual signature can be removed without destroying the useful signal?
- **Combining datasets:** use the larger hospital dataset PSG-Audio (287 patients) without biasing toward its acoustic conditions.
- **Night-level modeling:** apneas repeat; modeling the whole night instead of isolated 40 s segments.
- **Interpretability:** check that the model looks at silence and recovery gasps, not at background noise.

## Repository structure

```
├── README.md
├── rapport_apnee_sommeil.pdf
├── notebooks/
│   ├── 01_data_preparation.ipynb      # segmentation, mel-spectrograms, patient split
│   ├── 02_cnn_mel.ipynb
│   ├── 03_yamnet_transfer.ipynb
│   ├── 04_yamnet_adversarial.ipynb
│   ├── 05_cnn_bilstm.ipynb
│   └── 06_identity_probe.ipynb        # the 97.4% experiment
├── figures/
└── requirements.txt
```

## References

- Tao et al. (2025). Smartphone audio dataset for sleep apnea, *Nature Scientific Data*.
- Han et al. (2026). Transformer-based apnea detection, 40 s windows.
- Le et al. (2023). SoundSleepNet, noise-robust training.
- Wang et al. (2022). CNN-based apnea detection (OSAnet).

---

*Baptiste Wehry, ESILV (Data & AI). The report was written with the help of generative AI, based on experiments and analyses I carried out myself.*
