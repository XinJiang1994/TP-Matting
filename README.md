# TP-Matting
This is the official code for the paper "Privacy-Aware Real-Time Target Person Matting in Multi-Person Scenes Using Dual Encoder-Decoder Networks". TP-Matting is designed to perform real-time video matting for only the target person (the application user) only. It not only helps protect the environment privacy but also the person-related privacy. TP-Matting use a prepared reference image for and a vision transformer-based structure to locate the target person first. Then it use a CNN to refine the alpha matte. The proposed mehtod perform well on the target person matting, especially in the indoor scenes. And it can 56 FPS on an Nvidia 4090 GPU.

# News
- [Sep 10 2024] Inference code and pretrained models are published.
- [Sep 10 2024] Paper is submitted to The Visual Computer jurnal.

# How to use
## Install dependencies:
Install jittor following the tutorial: [Jittor Installation](https://cg.cs.tsinghua.edu.cn/jittor/download/) , then:

```sh
pip3 install -r requirements.txt
```

## Experience our meeting demo
Put your portrait photo in the VMDemoSys and configure the path in VMDemoSys/config.yml:
```sh
ui:
  w: 1366
  h: 960

ref:
  your_photo.jpg
```

Run the app:
```python
python3 app_entry.py
```

## Evaluation

# Dataset Download

# License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

