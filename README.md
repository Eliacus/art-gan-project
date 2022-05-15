# art-gan-project

## Downloading data

Two kaggle datasets are being used. To get the data, follow the steps below

1. Install kaggle: `pip install kaggle`
2. Download API credentials from your account page on kaggle and place in  `~/.kaggle/`
3. Join competition at: 'https://www.kaggle.com/competitions/gan-getting-started/overview'
4. Run the downloading script with: `bash download_data.sh`

## Data folder structure

Data folder structure should look like the tree example below, where all images are stored in images folder.

```bash
data/
├── art
│   └── images
└── celeba
    ├── images
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    └── list_landmarks_align_celeba.csv
```