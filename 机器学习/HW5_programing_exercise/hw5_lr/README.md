#### Data Description

---

* The dataset is a dataset of handwritten digits called MNIST[1], which contains contains a training set of 60000 examples and a test set of 10000 examples.

* Each image in this dataset  has 28*28 pixels and the associated label is the handwritten digit---that is, an integer from the set {0,1,....,9\}---in the image.

* **Loading the Dataset:** You can use the code below to  load the dataset automatically.  **Alternatively, you can also download the dataset through other methods/sources manually.**

  ```python
  from torchvision import datasets, transforms
  
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
  test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform
    )
  ```
  
  



[1] https://tensorflow.google.cn/datasets/catalog/mnist



