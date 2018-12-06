# Makefile
# Patrick Tung, Sylvia Lee (Nov 29, 2018)

# Description: This Makefile can be run to create our automatic
#							 data analysis pipeline.

# Usage:
#		To create the report: make all
#		To get a clean start: make clean

# Tags for long list of target/dependencies
CLEANDATA = data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv

FIGURES = results/figure/Age_plot.png results/figure/Sex_plot.png results/figure/Fare_plot.png \
			results/figure/Parch_plot.png results/figure/Pclass_plot.png results/figure/SibSp_plot.png

PREDICTIONS = results/train_prediction.csv results/test_prediction.csv

SUMMARIZATIONS = results/accuracies.csv results/feature_ranks.csv

# Run all analysis
all : docs/Titanic_Predictive_Data_Analysis.pdf
	rm -f results/figure/decision_tree
	rm -f docs/Titanic_Predictive_Data_Analysis.tex

# Clean raw data
$(CLEANDATA) : src/01_data_clean.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv
	python $^ $(CLEANDATA)

# Generate EDA visualizations
$(FIGURES) : src/02_data_exploratory_vis.py data/cleaned/cleaned_train.csv
	python $^ results/figure/

# Build decision classification tree and make predictions
$(PREDICTIONS) results/model/decision_tree_model.sav results/figure/CV_accuracy_score_lineplot.png : src/03_data_analysis.py $(CLEANDATA)
	python $^ results/

# Output results in presentation tables and figures
$(SUMMARIZATIONS) results/figure/decision_tree.png : src/04_data_summarization.py results/model/decision_tree_model.sav $(PREDICTIONS)
	python $^ results/

# Render report into PDF
docs/Titanic_Predictive_Data_Analysis.pdf : docs/Titanic_Predictive_Data_Analysis.Rmd $(PREDICTIONS) $(SUMMARIZATIONS) $(FIGURES)
	Rscript -e "rmarkdown::render('docs/Titanic_Predictive_Data_Analysis.Rmd')"

# Clean all output files generated
clean :
	rm -f $(CLEANDATA)
	rm -f $(FIGURES) results/figure/CV_accuracy_score_lineplot.png
	rm -f $(PREDICTIONS) results/model/decision_tree_model.sav
	rm -f $(SUMMARIZATIONS) results/figure/decision_tree.png
	rm -f docs/Titanic_Predictive_Data_Analysis.pdf docs/Titanic_Predictive_Data_Analysis.tex
