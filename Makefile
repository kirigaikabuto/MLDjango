all: run
.PHONY: all

run:
	cd ./machine_learning_model_api && python app.py
train_model:
	cd ./machine_learning_model_api/ml_model && python model_train.py
git:
	git add .
	git commit -m "feat:add update"
	git push origin master