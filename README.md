# Engine inspect

A computer vision system that checks if engine parts are assembled correctly.

# What is this project

When building car engines, there are small parts called bearing caps. 
Each bearing cap has a notch — a small V shaped mark. This notch must 
always point toward a specific hole in the engine block.

If even one notch points the wrong way — the part is defective and 
cannot be used.

This project builds a system that looks at a photo of the engine block 
and automatically says — is this part okay or not okay.

# The problem with doing this manually

On a real production line a new engine block arrives every few minutes. 
A human inspector has to check 4 notches on every single part. After 
hours of doing this — mistakes happen. Eyes get tired.

This system does the same check automatically in under one second.

# How it works

A deep learning model looks at the image and finds three things:
- The bolt holes in the engine block
- Notches that are pointing correctly toward the hole
- Notches that are pointing the wrong way
- 
If the model finds even one notch pointing the wrong way — the result 
is INVALID. If everything looks correct — the result is VALID.

# What I got wrong first

My first approach was to train the model to find notches — but not 
tell it which direction they should point. Then I wrote separate code 
to calculate the direction by measuring pixel brightness.

This worked fine when I tested it on my own images. But when I tested 
it on the actual production line — it kept giving wrong answers. 
Different lighting conditions, small shadows — all of this changed 
the pixel values enough to break the calculation.

The fix was simple but it took me a while to figure out.

Instead of teaching the model to find a notch and then calculating 
direction separately — I taught the model the direction directly.

I split the notch into two classes during labelling:
- notch pointing toward hole — correct
- notch not pointing toward hole — wrong

Now the model itself knows what correct and wrong looks like. No 
separate calculation needed.

# What I learned

The labelling is the most important part. Not the code.

When I labelled all notches the same way — the model had no idea about 
direction because I never showed it direction. The model can only learn 
what you teach it during training.

This sounds obvious after you figure it out. But it took seeing the 
wrong approach fail in a real production environment to understand it 
properly.

# Project files
scripts/
├── train.py          — trains the model on your images
├── predict_image.py  — runs prediction on a single image
├── predict.py        — prediction engine using ONNX format
├── verdict_rules.py  — decides VALID or INVALID
└── evaluate.py       — generates confusion matrix and metrics

dataset/
└── data.yaml         — tells the model where images are and 
                        what classes to look for

requirements.txt      — all Python libraries needed

# Classes the model detects

Class - What it means 
circle - The bolt hole in the engine block 
notch_toward_hole - Notch pointing correctly — good part 
notch_not_toward_hole  - Notch pointing wrong way — defective 

# How to set it up

Install the required libraries:
pip install -r requirements.txt

# How to train
python scripts/train.py --data dataset/data.yaml

Training takes around 1-2 hours depending on your machine.

# How to test on one image
python scripts/predict_image.py path/to/your/image.jpg --save

The result will be printed in the terminal and saved as an image 
with boxes drawn around each detected notch.

# How to evaluate the model
python scripts/evaluate.py

This generates:
- A confusion matrix showing where the model gets confused
- A bar chart showing precision, recall and F1 score per class
- Sample prediction images so you can see what the model detected

Results are saved in the results folder.

# Dataset

The images used to train this model belong to the manufacturing 
client and cannot be shared publicly.

The dataset has around 750 images taken on the production line. 
Images were labelled manually using MakeSense.ai.

# Built with

- YOLOv8 — object detection model
- OpenCV — image processing
- ONNX Runtime — faster inference in production
- Python 3.12
- Ultralytics

# Note

This project was built for a real manufacturing client. The company 
name is not shared # EngineInspect

A computer vision system that checks if engine parts are assembled correctly.

# What is this project

When building car engines, there are small parts called bearing caps. 
Each bearing cap has a notch — a small V shaped mark. This notch must 
always point toward a specific hole in the engine block.

If even one notch points the wrong way — the part is defective and 
cannot be used.

This project builds a system that looks at a photo of the engine block 
and automatically says — is this part okay or not okay.

# The problem with doing this manually

On a real production line a new engine block arrives every few minutes. 
A human inspector has to check 4 notches on every single part. After 
hours of doing this — mistakes happen. Eyes get tired.

This system does the same check automatically in under one second.

# How it works

A deep learning model looks at the image and finds three things:

- The bolt holes in the engine block
- Notches that are pointing correctly toward the hole
- Notches that are pointing the wrong way

If the model finds even one notch pointing the wrong way — the result 
is INVALID. If everything looks correct — the result is VALID.

# What I got wrong first

My first approach was to train the model to find notches — but not 
tell it which direction they should point. Then I wrote separate code 
to calculate the direction by measuring pixel brightness.

This worked fine when I tested it on my own images. But when I tested 
it on the actual production line — it kept giving wrong answers. 
Different lighting conditions, small shadows — all of this changed 
the pixel values enough to break the calculation.

The fix was simple but it took me a while to figure out.

Instead of teaching the model to find a notch and then calculating 
direction separately — I taught the model the direction directly.

I split the notch into two classes during labelling:
- notch pointing toward hole — correct
- notch not pointing toward hole — wrong

Now the model itself knows what correct and wrong looks like. No 
separate calculation needed.

# What I learned

The labelling is the most important part. Not the code.

When I labelled all notches the same way — the model had no idea about 
direction because I never showed it direction. The model can only learn 
what you teach it during training.

This sounds obvious after you figure it out. But it took seeing the 
wrong approach fail in a real production environment to understand it 
properly.

# Project files

scripts/
├── train.py          — trains the model on your images
├── predict_image.py  — runs prediction on a single image
├── predict.py        — prediction engine using ONNX format
├── verdict_rules.py  — decides VALID or INVALID
└── evaluate.py       — generates confusion matrix and metrics

dataset/
└── data.yaml         — tells the model where images are and 
                        what classes to look for

requirements.txt      — all Python libraries needed

# Classes the model detects

Classes and  What it means 
circle - The bolt hole in the engine block 
notch_toward_hole - Notch pointing correctly — good part 
notch_not_toward_hole - Notch pointing wrong way — defective 

# How to set it up

Install the required libraries:

pip install -r requirements.txt

# How to train

python scripts/train.py --data dataset/data.yaml

Training takes around 1-2 hours depending on your machine.

# How to test on one image

python scripts/predict_image.py path/to/your/image.jpg --save

The result will be printed in the terminal and saved as an image 
with boxes drawn around each detected notch.

# How to evaluate the model

python scripts/evaluate.py

This generates:
- A confusion matrix showing where the model gets confused
- A bar chart showing precision, recall and F1 score per class
- Sample prediction images so you can see what the model detected

Results are saved in the results folder.

# Dataset

The images used to train this model belong to the manufacturing 
client and cannot be shared publicly.

The dataset has around 750 images taken on the production line. 
Images were labelled manually using MakeSense.ai.

# Built with

- YOLOv8 — object detection model
- OpenCV — image processing
- ONNX Runtime — faster inference in production
- Python 3.12
- Ultralytics

# Note

This project was built for a real manufacturing client. The company 
name is not shared at their request.

AI assistance was used during development. The main lessons — 
especially understanding why the first approach was wrong — came 
from testing on the real production line and seeing it fail.
AI assistance was used during development. The main lessons — 
especially understanding why the first approach was wrong — came 
from testing on the real production line and seeing it fail.
