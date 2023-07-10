# Facial Expression Recognition using Convolutional Neural Network

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for Facial Expression Recognition. The model is trained on the FER2013 dataset, which consists of grayscale images of faces labeled with one of seven emotions: anger, disgust, fear, happy, neutral, sadness, and surprise.

## Requirements

- Python 3.x
- Pandas
- NumPy
- OpenCV (cv2)
- TensorFlow 2.x
- Matplotlib

## Dataset

The FER2013 dataset is used for training and evaluation. It contains 48x48 grayscale images of faces with corresponding emotion labels. The dataset is split into three subsets: training, validation, and test sets.

## Usage

1. Download the FER2013 dataset and save it as `fer2013.csv`.
2. Run the provided code in a Python environment with the required dependencies installed.
3. The code will preprocess the dataset, split it into training, validation, and test sets, and train a CNN model on the training data.
4. After training, the model's performance on the validation set will be displayed.
5. To classify a new image, use the `classify_image(image_path)` function, passing the path to the image file as the argument. The function will display the image and return the predicted emotion class.

## Model Architecture

The CNN model architecture consists of multiple convolutional layers with batch normalization, max pooling, and dropout for regularization. It is followed by fully connected layers for classification. The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

## Results

The training process will output the training and validation accuracy for each epoch. The final accuracy on the test set can be evaluated using the `evaluate()` function.

## Example Usage

```python
image_path = 'test.jpg'
predicted_class = classify_image(image_path)
print('Predicted Class:', predicted_class)
```

This will predict the image located at `test.jpg` and display the image along with the predicted emotion class.

## Acknowledgments

The code in this project is adapted from various open-source resources and tutorials on CNNs and facial expression recognition. The FER2013 dataset is publicly available and can be found online.

## References

- [FER2013 Dataset](https://www.kaggle.com/deadskull7/fer2013)
- [Convolutional Neural Networks (CNN) - TensorFlow Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Facial Expression Recognition using Convolutional Neural Networks](https://arxiv.org/abs/1710.07557)
