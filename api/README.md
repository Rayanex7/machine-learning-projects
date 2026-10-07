# Spam Filter API
A production-ready machine learning API built with FastAPI and scikit-learn, fully containerized with Docker.

## Tech Stack
* **Machine Learning:** Python, scikit-learn, TF-IDF Vectorization, Logistic Regression
* **Backend & Deployment:** FastAPI, Uvicorn, Docker

## How to Run (via Docker)
1. Clone this repository.
2. Build the image: `docker build -t spam-api .`
3. Run the container: `docker run -d -p 8000:8000 spam-api`
4. Open `http://127.0.0.1:8000/docs` to test the API.
