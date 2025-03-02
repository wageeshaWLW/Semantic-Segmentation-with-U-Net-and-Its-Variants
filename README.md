# 🏙️ Semantic Segmentation in Urban Scenes  

## 📌 Introduction  

Semantic segmentation is a core computer vision task that assigns a **class label to each pixel** in an image. This task is particularly **challenging in urban scene understanding** due to complex structures, varying lighting conditions, and occlusions.  

Deep learning has significantly advanced segmentation performance, with **U-Net and its extensions** being widely used due to their efficient **encoder-decoder architecture**. This project explores and compares the performance of three **segmentation models**:  

- **U-Net** 🏗 – A strong baseline with a simple yet effective architecture.  
- **Nested U-Net (U-Net++)** 🔗 – Uses **dense skip connections** to improve feature propagation.  
- **Attention U-Net** 🎯 – Incorporates **attention mechanisms** to enhance segmentation, particularly for fine details and occlusions.  

---

## 🎯 Objectives  

- Compare **U-Net, U-Net++, and Attention U-Net** on an **urban street dataset**.  
- Train models using a combination of **Cross-Entropy, Dice, and IoU losses**.  
- Evaluate segmentation accuracy in terms of **handling occlusions, fine details, and overall pixel-wise classification performance**.  
- Analyze the trade-offs between **model complexity and segmentation quality**.  
