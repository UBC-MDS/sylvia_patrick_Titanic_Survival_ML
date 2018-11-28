all : docs/Titanic_Predictive_Data_Analysis.pdf
	rm -f results/decision_tree

data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv : src/clean_data.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv
	python src/clean_data.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv

results/figure/Age_plot.png results/figure/sex.png results/figure/Fare_plot.png results/figure/Parch_plot.png results/figure/pclass.png results/figure/SibSp_plot.png : src/data_exploratory.py data/cleaned/cleaned_train.csv
	python src/data_exploratory.py data/cleaned/cleaned_train.csv results/figure/

results/test_prediction.csv results/train_prediction.csv results/model/decision_tree_model.sav : src/data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv
	python src/data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv results/

results/accuracies.csv results/feature_ranks.csv results/decision_tree.png : results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv src/summarize_data.py
	python src/summarize_data.py results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv results/

docs/Titanic_Predictive_Data_Analysis.pdf : docs/Titanic_Predictive_Data_Analysis.Rmd results/figure/Age_plot.png results/figure/sex.png results/figure/Fare_plot.png results/figure/Parch_plot.png results/figure/pclass.png results/figure/SibSp_plot.png results/test_prediction.csv results/train_prediction.csv results/accuracies.csv results/feature_ranks.csv
	Rscript -e "rmarkdown::render('docs/Titanic_Predictive_Data_Analysis.Rmd')"

clean :
	rm -f data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv
	rm -f results/figure/Age_plot.png results/figure/sex.png results/figure/Fare_plot.png results/figure/Parch_plot.png results/figure/pclass.png results/figure/SibSp_plot.png
	rm -f results/test_prediction.csv results/train_prediction.csv results/model/decision_tree_model.sav
	rm -f results/accuracies.csv results/feature_ranks.csv results/decision_tree.png
	rm -f docs/Titanic_Predictive_Data_Analysis.pdf docs/Titanic_Predictive_Data_Analysis.tex
