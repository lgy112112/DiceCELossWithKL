# 🧠 Enhanced Segmentation Loss Function

This repository contains the implementation of `DiceCELossWithKL`, an enhanced loss function for segmentation tasks that combines Dice loss, Cross-Entropy loss, and KL divergence to improve model performance on segmentation datasets.

## 🚀 Features

- **Dice Loss**: Measures the overlap between two samples.
- **Cross-Entropy Loss**: Computes the difference between two probability distributions for a given random variable or set of events.
- **KL Divergence**: Adds a regularization term to penalize the difference in probability distributions between the predictions and the ground truth.

## 🛠 Installation

To use this loss function, you need to have Python installed along with PyTorch and MONAI. It is easy to handle so I won't guide you here(sorry😊)

First, clone the repository using Git:

```bash
git clone https://github.com/lgy112112/DiceCELossWithKL.git
```

---

## 😎 Outcome Visualization
![image](https://github.com/lgy112112/DiceCELossWithKL/assets/144128974/6d9be282-4359-4d1b-b18e-1f7b197fc9f0)
The images respectively show the original image, the ground truth mask, the predicted mask by the KL-constrained model, and the predicted mask by the standard model. The first two rows clearly demonstrate the superiority of the KL-constrained model in shape control and over-segmentation control. It better fits the shape of the organs in the predictions. In contrast, the standard model, while predicting normal shapes, exhibits "hallucinations" or over-segmentation in blank areas, losing reference value. Similarly, in the third row, the standard model also shows over-segmentation of the gallbladder, whereas the KL model not only controls over-segmentation of this small class but also shows excellent shape control in complex segmentation scenarios.

Here's the translated table with the corresponding English explanation formatted for the README file:

## 🙌Comparison of Model Using Different Loss Performance

The following table shows the performance metrics of the KL-constrained model and the standard model for various organs. The images respectively show the original image, the ground truth mask, the predicted mask by the KL-constrained model, and the predicted mask by the standard model. The first two rows clearly demonstrate the superiority of the KL-constrained model in shape control and over-segmentation control. It better fits the shape of the organs in the predictions. In contrast, the standard model, while predicting normal shapes, exhibits "hallucinations" or over-segmentation in blank areas, losing reference value. Similarly, in the third row, the standard model also shows over-segmentation of the gallbladder, whereas the KL model not only controls over-segmentation of this small class but also shows excellent shape control in complex segmentation scenarios.

| Organ                     | KL-Constrained Model |                     | Standard Model |                     |
|---------------------------|----------------------|----------------------|----------------|----------------------|
|                           | F1                   | IoU                  | F1             | IoU                  |
| Spleen                    | 0.9850               | 0.9705               | 0.9362         | 0.8812               |
| Right Kidney              | 0.9877               | 0.9757               | 0.9445         | 0.8949               |
| Left Kidney               | 0.9839               | 0.9682               | 0.9741         | 0.9448               |
| Gallbladder               | 0.9203               | 0.8539               | 0.7904         | 0.6586               |
| Esophagus                 | 0.9344               | 0.8773               | 0.8939         | 0.8083               |
| Liver                     | 0.9871               | 0.9745               | 0.9672         | 0.9365               |
| Stomach                   | 0.9678               | 0.9376               | 0.9518         | 0.9086               |
| Aorta                     | 0.9865               | 0.9733               | 0.9898         | 0.9798               |
| Inferior Vena Cava        | 0.9563               | 0.9164               | 0.9027         | 0.8228               |
| Portal & Splenic Vein     | 0.7277               | 0.5723               | 0.5712         | 0.4035               |
| Pancreas                  | 0.9230               | 0.8572               | 0.7847         | 0.6536               |
| Right Adrenal Gland       | 0.8034               | 0.6741               | 0.7030         | 0.5588               |
| Left Adrenal Gland        | 0.9194               | 0.8528               | 0.7989         | 0.6725               |

The table above clearly shows that the KL-constrained model achieves higher F1 scores and IoU values for most organs, indicating better segmentation performance compared to the standard model.
