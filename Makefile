# Makefile
# Patrick Tung, Sylvia Lee (Nov 29, 2018)

# Description: This Makefile can be run to create our automatic
#							 data analysis pipeline.

# Usage:
#		To create the report: make all
#		To get a clean start: make clean

# Run all analysis
all : docs/Titanic_Predictive_Data_Analysis.pdf
	rm -f results/figure/decision_tree
	rm -f docs/Titanic_Predictive_Data_Analysis.tex

# Clean raw data
data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv : src/01_data_clean.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv
	python src/01_data_clean.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv

# Generate EDA visualizations
results/figure/Age_plot.png results/figure/Sex_plot.png results/figure/Fare_plot.png results/figure/Parch_plot.png results/figure/Pclass_plot.png results/figure/SibSp_plot.png : src/02_data_exploratory_vis.py data/cleaned/cleaned_train.csv
	python src/02_data_exploratory_vis.py data/cleaned/cleaned_train.csv results/figure/

# Build decision classification tree and make predictions
results/test_prediction.csv results/train_prediction.csv results/model/decision_tree_model.sav results/figure/CV_accuracy_score_lineplot.png: src/03_data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv
	python src/03_data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv results/

# Output results in presentation tables and figures
results/accuracies.csv results/feature_ranks.csv results/figure/decision_tree.png : results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv src/04_data_summarization.py
	python src/04_data_summarization.py results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv results/

# Render report into PDF
docs/Titanic_Predictive_Data_Analysis.pdf : docs/Titanic_Predictive_Data_Analysis.Rmd results/figure/Age_plot.png results/figure/Sex_plot.png results/figure/Fare_plot.png results/figure/Parch_plot.png results/figure/Pclass_plot.png results/figure/SibSp_plot.png results/test_prediction.csv results/train_prediction.csv results/accuracies.csv results/feature_ranks.csv
	Rscript -e "rmarkdown::render('docs/Titanic_Predictive_Data_Analysis.Rmd')"

# Clean all output files generated
clean :
	rm -f data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv
	rm -f results/figure/Age_plot.png results/figure/Sex_plot.png results/figure/Fare_plot.png results/figure/Parch_plot.png results/figure/Pclass_plot.png results/figure/SibSp_plot.png results/figure/CV_accuracy_score_lineplot.png
	rm -f results/test_prediction.csv results/train_prediction.csv results/model/decision_tree_model.sav
	rm -f results/accuracies.csv results/feature_ranks.csv results/figure/decision_tree.png
	rm -f docs/Titanic_Predictive_Data_Analysis.pdf docs/Titanic_Predictive_Data_Analysis.tex
