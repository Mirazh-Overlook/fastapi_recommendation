# fastapi_recommendation

To build recommendation system we actually need implicit data, since we do not have that yet we are using data from [here](https://www.kaggle.com/rmisra/news-category-dataset)

## How to use:

**To run the app in you computer:**

1. Clone the repo

```bash
git clone https://github.com/gauravchopracg/fastapi_recommendation.git
cd fastapi_recommendation/
```

2. Create and activate virtual environment
```bash
virtualenv venv
venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. download and unzip the dataset in the same folder

5. Run the app
```bash
uvicorn main:app
```
