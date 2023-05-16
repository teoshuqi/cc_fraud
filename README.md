# Dataset #
<https://www.kaggle.com/datasets/dermisfit/fraud-transactions-dataset>

# Set Up #
1. Create Virtual Environment
```bash
python3.8 -m venv venv
```
2. Activate Virtual Environment (for macos terminal)
```bash
source venv/bin/activate
```
3. Install requirements
```bash
pip3 install -r requirements.txt
```
4. Run MLFlow Server
```bash
mlflow server --backend-store-uri sqlite:///mydb.sqlite --host XX.XXX.XXX.XXX:XXXX
```
5. Try out model/Hyperparameter Tuning
```bash
python validate.py
```
6. Choose best hyperparameters and replace the values in ```best_params.json``` file
7. Train, log parameters/metrics, save and register model
```bash
python train.py
```
8. Test model and log metrics
```bash
python test.py
```
9. Serve model to expose it as an API
```bash
mlflow models serve -m "<model_uri>" -p <port> -h <host> --no-conda
```
10. Try out API
```bash
curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":[...],"data":[[...]]}}' http://XX.XXX.XXX.XXX:XXXX/invocations
```
