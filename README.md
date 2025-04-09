# PathMNIST CNN


## Model Sturcture And Project
This project is a assignment gived in Atılım University class CMPE452 which involved creating a Convolutional Neural Network (CNN)
to classify medical images from the PathMNIST dataset. This dataset contains
over 100,000 small images (28x28 pixels, 3 color channels) representing 9
different types of cancer.
The base model begins with basic data augmentation, including random
rotations and color adjustments. Initially, horizontal flipping was implemented,
but it negatively affected the model's performance, so it was removed. The CNN
consists of two main layers, each using 128 filters with a 3x3 kernel size,
followed by max pooling (kernel size 3, stride 2) and a 50% dropout rate. After
these layers, a fully connected layer with 512 neurons and another dropout layer
were added.
The model was trained for 20 epochs with a batch size of 128, using the
Adam optimizer and CrossEntropyLoss. An early stopping mechanism was
initially implemented but was later disabled after following the experiment
guidelines.
The dataset was split into training (89,996 images), validation (10,004),
and test (7,180) sets. These splits were provided by PathMNIST as separate
datasets.

![image](https://github.com/user-attachments/assets/7ac23ea4-7665-4cd2-ac7b-70825cdd1172)

## Experiments

Experiments were performed on 11 different hyperparameters as specified
in the guidelines. A significant portion of the experiments were performed and
logged automatically using my custom grid search program. For each
experiment, the training and validation losses, accuracies, and training duration
were recorded in JSON files. This automated logging system made visualizing
and analyzing the results more efficient.
The hyperparameters tested included:

- Kernel sizes of CNN (6 value)
- Dropout rates of CNN (6 value)
- Dropout rates of Classifier (6 value)
- Layers of CNN (6 value)
- Hidden neurons of Classifier (6 value)
- Max Pooling kernel sizes (6 value)
- Max Pooling strides (6 value)
- Epoch numbers (6 value)
- Batch sizes (8 value)
- Optimizers (3 optimizer type each with 2 difirent learning rate)
- Activation functions (3 function)
- Data Augmentation (only experimented on best model)

### 1. Kernel Size Experiments
![image](https://github.com/user-attachments/assets/fb9f259e-8328-44d6-97f5-df4c784c74ff)
![image](https://github.com/user-attachments/assets/a4742905-c7fc-4aee-8513-fafe2c1f6c88)


Generally, odd-numbered kernel sizes like 3×3 are preferred in CNNs. The
experiments tested less common kernels, such as 1×1, which showed
significantly worse performance. Kernel size selection is strongly related to input
dimensions. For smaller input sizes, such as the 28×28 images in the PathMNIST
dataset, it is easier to capture important features with smaller kernels. However,
even with small sized 2×2 (and other even-numbered kernels performs poorly),
the performance drop observed in the experimental results.
The base model uses standard 3×3 kernels, and experiments will continue
with this configuration in the best model, as it is clearly the top-performing
choice.

### 2. Dropout Rate Experiments

#### 2.1 CNN Dropout

![image](https://github.com/user-attachments/assets/14c335de-66ee-40ee-8f61-1d93a1112b5a)
![image](https://github.com/user-attachments/assets/7482b56d-606a-460a-b6c5-5247e87f5745)

The model applied dropout after each convolutional layer. The base model
initially used a 50% dropout rate. While dropout is effective for preventing
overfitting, high rates can lead to underfitting. The 50% dropout rate in the base
model appeared too aggressive, resulting in underfitting behavior.


Experimental testing showed 0% dropout achieved a decent accuracy
score. However, complete removal of dropout risks potential overfitting. Both
20% and 10% dropout provided better performance while maintaining
generalization. The 20% dropout rate’s validation accuracy scores began with
lower scores but eventually reached levels as high as the 10% dropout model.
Additionally, an important observation from the experiments is that in the 11th
epoch, the 50% dropout rate probably eliminated critical parts of the model,
causing a significant negative spike in performance during that epoch.
The primary concern here is model overfitting at low dropout rates. To
eliminate this risk, a 20% dropout rate appears safer to choose for the best
model.

#### 2.2 Classifier Dropout
![image](https://github.com/user-attachments/assets/8b6191e0-7782-4ca6-a182-699c319fac23)
![image](https://github.com/user-attachments/assets/02c5cf92-68cb-4f6f-84a5-091c753c75b2)

The model implements dropout only once between fully connected layers,
resulting in a more limited impact on the classifier compared to the CNN
sections. The base model implemented a 50% dropout rate in this configuration.
Experimental results showed slight improvement when using 30% dropout
rate and performance drop when using 20% in these layers compared to the
CNN’s dropout performance. Along with the earlier findings, the 30% dropout
rate gives optimal performance against 10% (to prevent overfitting) and chosed
for best model training.


### 3. Layers of CNN Experiments
![image](https://github.com/user-attachments/assets/b6ad577d-7a11-49ef-8170-bde63b38433a)
![image](https://github.com/user-attachments/assets/266b2879-faf7-40ce-ab47-9a4ff201c541)


The base model used 2 CNN layers with a fixed filter size of 128 (same size
for both layers). During experiments, each convolutional block was treated as a
single unit containing all components (dropout, max pooling, activation
functions, etc.). This approach caused an issue: when using more than 3 CNN
layers, max pooling reduced the image dimensions too much, resulting in errors.
To fix this, max pooling was limited to 3 layers and placed carefully. This
limitation may raise questions about experiment quailty. (The issue here was
using CNN’s layer padding as ‘same’ if a proper padding value is used this
wouldn’t be a problem. All of the experiments made with padding=’same’)
Adding or removing CNN layers showed little impact in experiments. A
better strategy might be testing different filter sizes per layer (like VGG models,
which increase filters incrementally). The best-performing models had 4 or 2
CNN layers, with very small differences between them. The validation graphs
gived very minimal advantage for 4 layers, so 4 layers chosed for the final model.

### 4. Hidden Neurons of Classifier Experiments
![image](https://github.com/user-attachments/assets/fddf6b93-ebd4-4d7b-abe9-bafef91ca143)
![image](https://github.com/user-attachments/assets/d1e6f43a-28b4-428d-a3de-8f07064846ec)


The base model used 512 neurons. Test results for 128, 512, and 1024
neurons were very close and it’s hard to pick a clear winner. Even though 512
and 1024 are much bigger than 128, their final scores were almost the same.
Surprisingly, using 8 times more neurons (like 1024 vs 128) didn’t affect
accuracy dramatically. This suggests changing neuron count doesn’t really help
here. At the 128 and 256 neuron counts model might reach the performance limit
which more neurons are unnecessary at this point. Also experiments shows that
256 neurons performed poorly compared to way smaller size 32. To keep things
simple and safe (safe is in terms of overfitting), neuron size is selected as 128.

### 5. Max Pooling Experiments

Max pooling normally implemented in each convolutional layer but limited
at max 3 application because dimesion issues. Explained in detail in CNN layer
count part.

#### 5.1 Max Pooling Kernel Size
![image](https://github.com/user-attachments/assets/94cffd3d-2638-411c-8632-f1e4474f8d66)
![image](https://github.com/user-attachments/assets/6ead520c-fc70-44c5-8797-7989d80fac0b)


The base model used a max pooling kernel size of 3. Generally, 2x2 and 3x
kernel sizes are used. A 2x2 kernel size is considered standard, while 3x3 is seen
as aggressive. Experiment results among these kernel sizes are very similar to
each other, except for 1x1, which does not apply pooling.
Experimental results show that the best value is obtained with a 4x4 kernel
size. However, this size is not particularly useful (and very rarely used) and did
not demonstrate strong generalization. Thus, sticking with the 3x3 kernel size
remains optimal for the best model. On the other hand, despite being the
standard choice, the 2x2 kernel size’s performance is less reliable, as its graphs
include several sharp spikes.

#### 5.2 Max Pooling Stride
![image](https://github.com/user-attachments/assets/6f670c87-fb04-478f-8591-dd105a471a41)
![image](https://github.com/user-attachments/assets/50399098-cc7f-4770-b173-2c33700c2c8a)

The most common value for stride is “2”, as used in the base model. This
means downsampling the resolution by half at each step.
Experiments clearly show that a stride of 1 would be a very poor choice. On
the other hand, stride values of 2 and 3 performed very well. Similar to how a
kernel size of 3 is considered a more aggressive approach for stride selection,
stride 3 can sometimes lead to loss of features, but it showed a slight advantage
over stride 2 in these experiments. To keep this minor advantage, stride 3 was
selected for training the best model.


### 6. Epoch Number Experiments
![image](https://github.com/user-attachments/assets/cc34f5ed-a847-4fef-ae9e-4300f3c0d2f2)
![image](https://github.com/user-attachments/assets/3122ce01-d966-4f07-a17c-c244b1626fd9)

The epoch count was specified as 20 for the base model. Unnecessarily
large epoch counts often lead to overfitting, while low epoch counts typically
result in underfitting.
Experiments show that 10 or 15 epochs are not enough for optimal
training. Additionally, after the 20th epoch, no significant improvement can be
observed. Based on these findings, the ideal epoch count is selected as 20. If any
overfitting issues occures reducing epochs to the 15 would be best.
**Tests with Best Model:** Best model trained with 25, 20, and 15 epochs
individually and least overfitting result came with 15 epoch.

### 7. Batch Size Experiments
![image](https://github.com/user-attachments/assets/355d5329-5bca-4077-8c08-236d9fba2b4c)
![image](https://github.com/user-attachments/assets/35aed187-afe8-445b-9bc7-22a6ed2748f7)



Initiatlly Batch size selected randomly 128 for the base model. Actually I
didn’t know it was so important for model. Turns out batch size is important both
for model performance and computational cost.
According to experiment results; while large batch sizes like 1024, 512,
and 256 negatively affects the model performance, low batch sizes gives more
convenient and similar results such as 16, 32, and 64. However mini-batches like
4 and 8 gives slightly more poor results but they are still better than large ones.
From this findings batch size 16, 32, and 64 should be tested more for best
model.
**Tests with Best Model:** Best model trained each of this values
individually and best value decided as 64 in terms of test accuracy scores. Test
accuracy scores:

- Batch size 64: %89
- Batch size 32: %84
- Batch size 16: %81

### 8. Optimizer Type Experiments

![image](https://github.com/user-attachments/assets/7a847d1b-c869-497b-85e7-1e6218fae903)

![image](https://github.com/user-attachments/assets/94a684a4-21f6-466a-bfcd-a2abfea4bee9)


The Adam optimizer is the most popular and simpler selection among all
optimizers. So, the base model used the Adam optimizer with a 0.0001 learning
rate. In truth, the learning rate alone deserves its own experiment, but tests
were conducted for 0.0001 and 0.001 with Adam, RMSprop, and SGD.
Momentum was set to 0.9 for RMSprop and SGD.
These experiments resulted in chaotic outcomes. Optimizer selection is a
serious matter requiring careful tuning. Due to this complexity and limitations in
technical skills of mine, all experiments except the base model’s configuration
failed. Therefore, Adam with a 0.0001 learning rate will continue to be used.

### 9. Activation Function Experiments

![image](https://github.com/user-attachments/assets/d061b829-a187-4c11-ab94-2e19f6db3399)

![image](https://github.com/user-attachments/assets/cbaf2a16-b80e-467f-b3b2-f5f60c1ada7a)

The base model uses the ReLU activation function in each convolutional
and classifier layer.
The experiment results clearly show that ReLU performs best compared to
Tanh and Sigmoid. Sigmoid activation works best for probabilities in cases with
fewer classes, but this dataset includes 9 classes, which is why it performs
poorly. Because of reliability and good performance ReLU will continue to be
used for training the best model.


### 10. Data Augmentation

![image](https://github.com/user-attachments/assets/ac569782-818b-4516-99b6-84d8f784596b)

![image](https://github.com/user-attachments/assets/fe4ee8ab-7905-47a8-8623-d04ff82a3bf2)

In early stages of my experiments data augmentation seemed as a good
idea but when things comes to practice, it was affecting in bad way. I wanted to
demonstrate this the difference on best model. Here are the augmentations used:

- RandomHorizontalFlip: Flips image vertically.
- RandomRotation: Randomly rotates the image respect to given parameters.
- ColorJitter: Randomly adjusts brightness, contrast, saturation, and hue.

## Train Durations

![image](https://github.com/user-attachments/assets/a651a4c9-cdb5-4aa6-a926-01061fbead70)

![image](https://github.com/user-attachments/assets/574a5afc-5403-4fc0-8b35-79df9acf8010)

Total training time for all experiments took: 6 and half hours. (Trains are made in
a personal computer so the result times are not isolated from other uses of pc.)



## Best Model

According to all of the experiments, best resulting hyper parameters are
listed:

- Kernel size of CNN : 3x
- Dropout rate of CNN : %
- Dropout rate of Classifier : %
- Layers of CNN : 4
- Hidden neurons of Classifier : 128
- Max Pooling kernel size : 3
- Max Pooling stride : 3
- Epoch number : 15
- Batch size : 64
- Optimizer : Adam with lr=0.
- Activation function : ReLU

Best Model Architecture:<br>
![image](https://github.com/user-attachments/assets/01144147-0224-419d-90ac-3de040bd0def)


### Evaluation

#### Train - Validation Metrics

![image](https://github.com/user-attachments/assets/b27579a4-9ae6-4d46-92ae-bba20eedee5c)

![image](https://github.com/user-attachments/assets/a3ec1b5a-c693-48dc-8006-ad6bef5f25cb)

Best model trained in 3 minutes 50 seconds.Best model performed better
from all previous experiments in terms of validation metrics. Validation graphs
are nearly linear to train graphs which is a sign of good learning. There is no
sharp spikes which also a good sign. Validation accuracy goes decently high up
to %95.8 with mean of %91.1.

#### Test Results

```
Best Models performance on the test split:
```
- **Test Accuracy:** %89.05
- **Test Loss:** 0.4888
- **Precision:** %89.34
- **Recall:** %89.05
- **F1-Score:** %88.82
    Best Model’s general performance can be considered as good except for 2nd^
and 7th classes. This difference might caused by imbalanced classes or just
models issue. You can see the classes individual test scores from classification
report and confussion matrix:

![image](https://github.com/user-attachments/assets/568dcf36-73fb-46e0-8848-cc99ce4a8cd9)

![image](https://github.com/user-attachments/assets/d5705a6e-3a27-47c8-aef0-202e7e937ed2)



## Conclusion

Experimenting all hyperparameters on some model, provides really good
understanding about each hyperparameter’s duty and effect to overall. With this
approch you are not leaving anything to luck. It didn’t surprised me to obtain
nearly 90% test accuracy because of my detailed and carefull work.
I gained a strong understanding about how batch size effects the models
outcomes. Low and High batch sizes became a disadvantage and ı should have
find the optimal value. I realized increasing dropout rates higher than 30% leads
to underfitting and loss of important features. The next very important learning
of mine was max pooling and using it without proper padding. When more than 3
convolution layer and max pooling added, model gived warnings about the input
dimensions. It was becoming so small because of the lack of padding. I overcome
this issue by not using more than 3 max pooling.
The findings showed that you should’t consider a parameter value as best if
it gives highest validation accuracy or lowest loss. You really be carefull about
overfitting. For instance using 20 or 25 epochs seemed reasonable for base
model but when training the best model it was clear that model started to
memorize after 15th epoch. So it is a good practice to never stop experimenting.
It is advised to do more experiments with considering this best model as
base model because each hyper parameter works in a corolation. They might
give better results with other parameters and their -recently titled as- “worse”
values.


