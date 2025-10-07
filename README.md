# BrainCSD

**BrainCSD** is a three-stage framework for multimodal brain connectome synthesis and refinement, supporting both fMRI (functional) and dMRI (structural) inputs. The pipeline consists of:

1. **ROI Activation**  
2. **Encoding with Mixture-of-Experts (MoE)**  
3. **Network-Aware Finetuning**

All dependencies are listed in `requirements.txt`.

---

## 📦 Stage 1: ROI Activation

In this stage, a Transformer-based encoder-decoder learns modality-specific representations aligned with brain atlases.

- Configure Transformer hyperparameters (e.g., `dim`, `mlp_dim`, `depth`) in the ROI activation scripts (`roi_activation/`).
- **Important**: Use **different configurations for fMRI and dMRI**, as their spatial resolutions and signal characteristics differ significantly.
- Due to the inherently low signal strength in atlas × dMRI features, **scale the dMRI input by a factor of 10** before feeding it into the model.
- Save the trained weights—they will be loaded in Stages 2 and 3.

---

## ⚙️ Stage 2: MoE-Based Encoding

This stage models dynamic or cross-modal interactions using a Mixture-of-Experts (MoE) architecture.

- **Start by loading the pretrained model from Stage 1.**
- The number of experts per MoE component can be customized directly in `models/connect_gen.py` (under the `encode/` directory).
- Note: Expert counts are **not controlled via command-line arguments** (i.e., not in `opts.py`)—they must be set in the source code for flexibility and reproducibility.

---

## 🔧 Stage 3: Network-Aware Finetuning

The final stage refines the preliminary functional (FC) or structural (SC) connectomes using network-informed MoEs.

- Load pretrained weights from **both Stage 1 and Stage 2**.
- Specify the refinement mode: `--refine-mode FC` or `--refine-mode SC`.
- **Tip**: The optimal performance depends on consistent embedding dimensions across stages. Ensure the Transformer embedding settings (e.g., `dim`, `depth`) match those used in earlier stages unless intentionally fine-tuning them.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Stage 1 (example for fMRI)
python train_roi.py --modality fmri --scale-dmri 1.0

# Run Stage 2
python train_encode.py --load-roi-ckpt ./checkpoints/roi_fmri.pth

# Run Stage 3 (FC refinement)
python finetune.py --load-encode-ckpt ./checkpoints/encode.pth --refine-mode FC
