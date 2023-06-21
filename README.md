# Classification-of-Diseases-in-K-melon
Deep Learning Team project in Sookmyung university\
The damage caused by K-melon diseases is serious. Here's how to solve this problem efficiently.

### Learning Environment
Colab Pro

## Main features of the data
* Data source : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100
#### [Data proprocessing](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/blob/main/Data_preprocessing.ipynb)
1. Delete images taken at night (day section = '야간')
2. Split Test, Train data -Test data is FM-02,FM-04 data (FM-ID = FM-02,FM-04)
3. '흰가루병유사' Data augmentation
4. Split Train, Validation data
---
* Data pipeline
```python
!wget https://www.dropbox.com/s/diwzbpumn81dwhd/img4.zip?dl=0
```
Download and Unzip file

```python
!unzip '/content/img4.zip?dl=0'
```
|Downy mildew|Powedery mildew|Normal|
|:---:|:---:|---|
|![노균병](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/fc85bd20-7ad4-4ebf-9f72-9143a4c1747c)|![흰가루병](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/624487cd-92c9-44d4-bb5a-8676e8301cc8)|![S027-FM02-027-2022-07-07-000070](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/e4f96ccd-bcda-47aa-bc62-9633ad233b7d)|
|yellow, yellowish brown spots|white spots|

## Data generator
```python
epochs = 50
batch_size = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
```
**Data augmentation** (shear, brightness, flip)
```python
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                          shear_range = 60,                                                                                                                    # shear_range = 60,
                                                          brightness_range = (0.8,1.2),
                                                          horizontal_flip = True)
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
```
**Data generator** (train, validation, test)
```python
train_set = train_gen.flow_from_directory(directory="/content/img4/train",
                                        target_size = (IMG_WIDTH,IMG_HEIGHT),
                                        class_mode='categorical',
                                        batch_size = batch_size)
val_set = img_gen.flow_from_directory(directory="/content/img4/val",
                                        target_size = (IMG_WIDTH,IMG_HEIGHT),
                                        class_mode='categorical',
                                        batch_size = batch_size)
test_set = img_gen.flow_from_directory(directory="/content/img4/test",
                                        target_size = (IMG_WIDTH,IMG_HEIGHT),
                                        class_mode='categorical',
                                        batch_size = batch_size)
```
## Model comparison
|Model|Confusion matrix|Accuracy|
|------|---|---|
|[Transfer learning-Resnet50](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/blob/main/ResNet50_transferlearning.ipynb)|<img width="385" alt="trns-res" src="https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/f60ae42d-a4af-4856-8669-15feafdddbf1">|83.7%|
|[Transfer learning-VGG16](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/blob/main/VGG16_Transfer.ipynb)|<img width="385" alt="trans-vgg" src="https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/63eab467-bbcc-4bb3-a440-20bb68925962">|81.3%|
|[Semi-supervised learning-Resnet50](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/blob/main/semi_ResNet50.ipynb)|<img width="385" alt="semi-res" src="https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/191c83f8-fc58-4af4-9197-a1a95126967b">|55.7%|
|[Semi-supervised learning-VGG16](https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/blob/main/semi_Vgg16.ipynb)|<img width="385" alt="semi-vgg" src="https://github.com/YooJung-Moon/Classification-of-Diseases-in-K-melon/assets/89197996/4d0fa5b6-fa2c-48c6-9271-9b6525453933">|82%|
## Conclusion
Transfer learning-Resnet50 Model is **Best**!!
