# Daxmod

Daxmod is a **python toolbox** designed for, but not limited to, the simplification of text classification tasks.

Daxmod includes modules for data loading, feature extraction, feature selection, model building and evaluation, and object persistence.

![daxmod_package](https://user-images.githubusercontent.com/45699578/174140781-92de933c-0d1a-48c7-8853-9cec77fac81d.svg)

___

## Working with Daxmod

Daxmod supports two main workflows: the **building workflow** and the **pretrained workflow**.

![daxmod_workflow](https://user-images.githubusercontent.com/45699578/174140860-a7f85f97-cc84-4bad-925f-2204f86982b8.svg)

The **building workflow** performs text classification through several steps while the **pretrained workflow** use pre-trained models.

### Access

Daxmod can load two types of datasests: data in raw format and in tabular format.

#### Load data in raw format

Parmaeters:

- **dataset_folder**: the path of the dataset folder
- **sub**: the subset to load. Set to `train` by default
- **encoding**: the encoding of the file. Set to `utf-8` by default
- **n_jobs**: the number of processors to use to load the file. Set to `None` by default.
  - `None` uses one (1) processor
  - `2` uses two (2) processors
  - `-1` uses all the processors available on the machine

```python
# Import load_from_folder from the access module
from daxmod.access import load_from_folder

# Load the train data
train_data = load_from_folder(dataset_folder='imdb', n_jobs=-1)
```

#### Load data in tabular format

Parameters:

- **filepath**: the path of the file
- **header**: specify if the file has a header. `infer` is used by default. Use `None` when there is no header.
- **sep**: the separator used for delimitation. `,` is used by default.
- **encoding**: the encoding of the file. Set to `utf-8` by default

```python
# Import load_from_file from the access module
from daxmod.access import load_from_file

# Load the data
data = load_from_file(filepath='imdb.csv')
```

### Extraction

Predefined classification algorithms availalbe in Daxmod are:

- **bow** (Bag of Words)
- **bigrams** (Bi-grams)
- **trigrams** (Tri-grams)
- **tf** (Term Frequency)
- **tf-idf** (Term Frequency-Inverse Document Frequency)

This code snippet shows how to extract features using the bag of words method.

```python
# Import Extractors class from the extractor module
from daxmod.extraction import Extractors

# Instantiate a bag of words extractor
extractor = Extractors(extractor='bow')

# Fit the extractor and transform the data
extractor.fit(X, y)
X_transformed = extractor.transform(X)
```

We can also create custom N-grams extractors with Daxmod.

This code snippet shows how to create a custom N-grams extractor.

```python
# Import Ngrams to define custom extractor
from daxmod.predefined.extractors import Ngrams

# Create custom N-grams with unigrams and bigrams
extractor = Ngrams(ngram_range=(1,2))

# Fit and transform the data with the created Ngrams
extractor.fit(X, y)
X_transformed = extractor.transform(X)
```

### Selection

Predefined selection algorithms availalbe in Daxmod are:

- **anova** (Analysis of Variance)
- **chi2** (Chi-squared)

This code snippet shows how to select features using the ANOVA method.

```python
# Import the SelectTopK class from the module selection
from daxmod.selection import SelectTopK

# Use ANOVA as selection method and 10K as the number of features to keep
topk = SelectTopK(score_func='anova', k=10000)

# Fit and select the best features
topk.fit(X_transformed, y)
X_top = topk.transform(X_transformed)
```

### Classifiers

#### Models

Predefined classification algorithms availalbe in Daxmod are:

- **mlp** (Multi-Layer Perceptron)
- **naive_bayes** (Naive Bayes)
- **svm** (Support Vector Machine)

This code snippet shows how to train a classifier. We used a predefined SVM classifier available in Daxmod.

```python
# Import the Models class from daxmod
from daxmod.classifiers.models import Models

# Instantiate an SVM classifier
model = Models(classifier='svm')

# Train the SVM classifier
model.fit(X_top, y)
```

Daxmod provides an evaluation method for the trained models.

The available metrics are:

- **accuracy**
- **balanced_accuracy**
- **f1**
- **precision**
- **recall**

```python
# Evaluate the model on the test set (accuracy)

model.evaluate(X_test_top, y_test, metric='accuracy')
```

Daxmod provides the *predict* method to make predictions on a set of data.

```python
# Make prediction with the trained model

y_pred = model.predict(X_new)
```

Daxmod can also use classification algorithms imported from scikit-learn.

```python
# Import RandomForest from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Pass an instance of the RandomForest as the parameter to Models
model = Models(classifier=RandomForestClassifier())
```

#### Pretrained

Daxmod only includes a **BERT** pre-trained model currently.

This code snippet shows how to use the pretrained BERT.

```python
# Import Bert from pretrained module
from daxmod.classifiers.pretrained import Bert

# Instantiate bert for a binary classifier
bert = Bert(n_classes=2)

# Autocompile the model 
bert.auto_compile()

# Train the model
bert.fit(X, y)

# Evaluate the model
bert.evaluate(X_test, y_test)
```

### Persistence

This code snippet shows to how to save and load objects with Daxmod.

```python
# Save a model without the persistence module
model.save(name='svm', folder='models')

# Import the method save_object from the persistence module
from daxmod.persistence import save_object

# Save a model in the folder 'models' with 'svm' as name
save_object(obj=model, name='svm', folder='models')

# Import the method load_object from the persistence module
from daxmod.persistence import load_object

# Load a saved model
model = load_object(path='models/svm.model')
```

___

Daxmod includes a module called **utils**, which provides numerous functions to perform rapid actions on the data. Here are several of them:

- **feature_label_split** to split the loaded data into feature and label.

```python
from daxmod.utils import feature_label_split

X, y = feature_label_split(train_data)
```

- **encode_labels** to encode labels in string format.

```python
from daxmod.utils import encode_labels

y_encoded = encode_labels(y)
```

- **count_labels** to count the number of labels in a dataset.

```python
from daxmod.utils import count_labels

n_labels = count_labels(train_data)
```

___

## Installation

### Dependencies

Daxmod requires:

- Python (>=3.8)

- joblib (>=1.1.0)

- numpy (>=1.21.4)

- pandas (>=1.3.4)

- scikit-learn (>=1.0.1)

Optional dependencies. Required for pretrained models:

- tensorflow (>=2.7.0)

- tensorflow_hub (>=0.12.0)

- tensorflow_text (>=2.7.3)

### User installation

#### Installation without pretrained dependencies

```bash
pip install -U daxmod
```

#### Installation with pretrained dependencies

```bash
pip install -U daxmod[pretrained]
```

### Source code

```bash
git clone https://github.com/Authentic10/daxmod.git
```
