


python3 src/main.py -model decisiontree 2>&1 | tee decisionTreeBinary.txt;
python3 src/main.py -model knn 2>&1 | tee knnBinary.txt;
python3 src/main.py -model mlp 2>&1 | tee mlpBinary.txt;
python3 src/main.py -model svc 2>&1 | tee svcBinary;
python3 src/main.py -model randomforest 2>&1 | tee randomforestBinary.txt;
python3 src/main.py -model catboost 2>&1 | tee catboostBinary.txt;
python3 src/main.py -model xgboost 2>&1 | tee xgboostBinary.txt;
python3 src/main.py -model none 2>&1 | tee noneBinary.txt;
python3 src/main.py -model histgradientboosting 2>&1 | tee histgradboostingBinary.txt;
python3 src/main.py -model gradientboosting 2>&1 | tee gradboostingBinary.txt;


python3 src/main.py -default -balancedata smoteenn; python3 src/main.py -default -balancedata smotetomek; python3 src/main.py -default -balancedata randomundersampler; python3 src/main.py -default -balancedata tomeklinks; python3 src/main.py -default -balancedata smoten; python3 src/main.py -default -balancedata adasyn; python3 src/main.py -default -balancedata kmeanssmote;

python3 src/main.py -default -balancedata smoteenn -multiclass; python3 src/main.py -default -balancedata smotetomek -multiclass; python3 src/main.py -default -balancedata randomundersampler -multiclass; python3 src/main.py -default -balancedata tomeklinks -multiclass; python3 src/main.py -default -balancedata smoten -multiclass; python3 src/main.py -default -balancedata adasyn -multiclass; 

python3 src/main.py -balancedata randomundersampler | tee optimizeBinaryRUndersampler.txt; python3 src/main.py -balancedata smoteenn | tee optimizeBinarySmoteenn.txt; python3 src/main.py -balancedata tomeklinks | tee optimizeBinaryTomekLinks.txt; 

python3 src/main.py -multiclass -balancedata randomundersampler | tee optimizeMulticlassRUndersampler.txt; python3 src/main.py -multiclass -balancedata smoteenn | tee optimizeMulticlassSmoteenn.txt; python3 src/main.py -multiclass -balancedata tomeklinks | tee optimizeMulticlassTomekLinks.txt;



#Experiments with less hyperparams:
#Binary:
python3 src/main.py -lessparams -balancedata randomundersampler | tee optimizeBinaryRUndersamplerLess.txt; python3 src/main.py -lessparams -balancedata smoteenn | tee optimizeBinarySmoteennLess.txt; python3 src/main.py -lessparams -balancedata tomeklinks | tee optimizeBinaryTomekLinksLess.txt; python3 src/main.py -lessparams | tee optimizeBinaryNoneLess.txt; 
#Multiclass:
python3 src/main.py -multiclass -lessparams -balancedata randomundersampler 2>&1 | tee optimizeMulticlassRUndersamplerLess.txt; python3 src/main.py -multiclass -lessparams -balancedata smoteenn 2>&1 | tee optimizeMulticlassSmoteennLess.txt; python3 src/main.py -multiclass -lessparams -balancedata tomeklinks 2>&1 | tee optimizeMulticlassTomekLinksLess.txt; python3 src/main.py -multiclass -lessparams 2>&1 | tee optimizeMulticlassNoneLess.txt; 




