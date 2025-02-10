Semantic segmentation is a fundamental task in
computer vision, aiming to assign a class label to each
pixel in an image. This problem is particularly challenging
in urban scene understanding, where complex
structures and varying lighting conditions impact segmentation
accuracy. Deep learning has significantly
improved performance in this domain, with U-Net
and its extensions being widely used due to their efficient
encoder-decoder architecture.
In this project, we explore and compare the performance
of three segmentation models: U-Net, Nested
U-Net (U-Net++), and Attention U-Net. These architectures
are evaluated on an urban street dataset
with pixel-wise annotations. The models are trained
using a combination of Cross-Entropy, Dice, and IoU
losses to optimize segmentation performance.
Our experiments show that while the standard
U-Net provides a strong baseline, Nested U-Net enhances
feature propagation through dense skip connections,
and Attention U-Net further improves segmentation
by leveraging attention mechanisms. The
results indicate that Attention U-Net achieves superior
segmentation accuracy, particularly in handling
occlusions and fine details. The findings highlight the
strengths and limitations of each model and provide
insights into their applicability for real-world semantic
segmentation tasks.
